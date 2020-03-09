import torch as torch
from torch import nn as nn
import torch.nn.utils.rnn as torch_utils
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder Network
    
    Arguments:
        torch {nn.Module} -- Inherits from nn.Module
    
    Raises:
        NotImplementedError: Only RNNs of type LSTM and GRU are allowed. Otherwise this error this thrown.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0, batch_first=True, rnn_type='LSTM'):
        """Constructor
        
        Arguments:
            input_size {int} -- number of features per time step
            hidden_size {int} -- number of hidden nodes per time step
            num_layers {int} -- number of layers
        
        Keyword Arguments:
            dropout {float} -- percentage of nodes that should switched out at any term (default: {0})
            batch_first {bool} -- if shape is (batch, sequence, features) or not (default: {True})
            rnn_type {str} -- which type of rnn cell should be used (default: {'LSTM'})
        
        Raises:
            NotImplementedError: Only RNNs of type LSTM and GRU are allowed. Otherwise this error this thrown.
        """
        super(Encoder, self).__init__()

        # number of features at a given time step
        self.input_size = input_size
        # number of hidden units (at this time step)
        self.hidden_size = hidden_size
        # number of layers
        self.num_layers = num_layers

        # select the rnn_type connection
        if rnn_type == 'LSTM':
            self.model = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers,
                batch_first=batch_first,
                dropout=dropout
            )
        elif rnn_type == 'GRU':
            self.model = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers,
                batch_first=batch_first,
                dropout = dropout
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        """Foreward pass
        
        Arguments:
            x {PackedSequence} -- Batch input, having variational length.
        """
        # Passing in the input and hidden state into the model and obtaining outputs
        h_t, (h_end, c_end) = self.model(x)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # h_end.shape[1] -> batch size
        # h_end.shape[2] -> hidden size
        # creates the output that fits
        h_end_out = h_end[-1,:,:]
        #out = self.linear_adapter(lin_input)
        return(h_t, (h_end, c_end), h_end_out)


class Variational(nn.Module):
    """Variation Layer of Variational AutoEncoder
    
    Arguments:
        torch {nn.Module} -- Inherits from nn.Module
    """
    def __init__(self, hidden_size, latent_size, use_identity=False):
        """Constructor
        
        Arguments:
            hidden_size {int} -- number of features per time step (output from encoder)
            latent_size {int} -- what size the latent vector should be
        """
        super(Variational, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.use_identity = use_identity

        # define linear adapter from h_end to mean, logvar
        if self.use_identity:
            
            # NOTE: that if you use identity transformation, 
            # the output of this layer will be showing the 
            # same dimensionality as the input.

            # use identity transformation
            self.hidden_to_mu = nn.Identity()
            self.hidden_to_tanh = nn.Linear(self.hidden_size, self.latent_size)
            self.act_tanh = nn.Tanh()
            self.than_to_exp = nn.Linear(self.hidden_size, self.latent_size)
            self.act_elu = nn.ELU()
 
        else:

            self.hidden_to_mu = nn.Linear(self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_size)

            # Fills the input Tensor with values according to the method described in 
            # Understanding the difficulty of training deep feedforward neural networks - 
            # Glorot, X. & Bengio, Y. (2010)
            nn.init.xavier_uniform_(self.hidden_to_mu.weight)
            nn.init.xavier_uniform_(self.hidden_to_logvar.weight)


    def forward(self, hidden, mask=None):
        """Forward pass
        
        Arguments:
            hidden {torch.tensor} -- last hidden state of the Encoder
        """
        # get mu and logvar
        self.mu = self.hidden_to_mu(hidden)
        # activation layer
        if self.use_identity:
            self.logvar = self.hidden_to_tanh(hidden)
            self.logvar = self.act_tanh(self.logvar)
            self.logvar = self.than_to_exp(self.logvar)
            self.logvar = self.act_elu(self.logvar)
        else:
            self.logvar = self.hidden_to_logvar(hidden)

        # calculate std        
        std = self.logvar.mul(0.5).exp_()
        # get epsilon based on normal distribution
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        
        eps = torch.autograd.Variable(eps)
        # return latent output
        latent = eps.mul(std).add(self.mu)

        # mask output
        if mask is not None:
            latent = latent.masked_fill(mask == 0, 0)
        
        return(latent)


class MultiHeadAttention(nn.Module):
    """Multihead self-attention
    
    Arguments:
        nn {nn.Module} -- Inherits from nn.Module
    
    Returns:
        (x, attention) -- x = attended output, attention = attention_weights
    """
    def __init__(self, hidden_size, n_heads, dropout, device):
        """Constructor
        
        Arguments:
            hidden_size {int} -- hidden size of the input
            n_heads {int} -- number of heads (attention heads)
            dropout {double} -- dropout rate
            device {string} -- cuda or cpu
        """
        super().__init__()
        
        assert hidden_size % n_heads == 0
        
        # init values
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        
        # linear layers
        self.hidden_to_query = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_key = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_value = nn.Linear(hidden_size, hidden_size)
        
        # potentially remove
        self.z_to_output = nn.Linear(hidden_size, hidden_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # scale
        self.scale = torch.sqrt(torch.FloatTensor([self.head_size])).to(device)
        
    def forward(self, query, key, value, mask = None):
        """Forward pass
        
        Arguments:
            query {tensor} -- query tensor
            key {tensor} -- key tensor
            value {value} -- value tensor
        
        Keyword Arguments:
            mask {mask} -- optional mask for variable length input (default: {None})
        
        Returns:
            (x, attention) -- x = attended output, attention = attention_weights
        """
        # get batch size
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
    
        # transform query, key and value
        Q = self.hidden_to_query(query)
        K = self.hidden_to_key(key)
        V = self.hidden_to_value(value)

        # mask output, to prevent wrong energy calculation
        #if mask is not None:
        #    Q = Q.masked_fill(mask == 0, 0)
        #    K = K.masked_fill(mask == 0, 0)
        #    V = V.masked_fill(mask == 0, 0)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        
        # reorder dimensions
        Q = Q.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
        
        # calculate energy
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, seq len, seq len]
        
        # mask energy and attention 
        if mask is not None:
            energy_mask = mask.view(batch_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
            # energy_mask = [batch size, n heads, query len, head dim]
            energy_mask = torch.matmul(energy_mask, energy_mask.permute(0, 1, 3, 2)) == 0
            # energy_mask = [batch size, n heads, seq len, seq len]
            energy = energy.masked_fill(energy_mask, -1e10)
            # masked input for attention
            attention = torch.softmax(energy, dim = -1)  
            # attention -> not masked on rows
            attention = attention.masked_fill(energy_mask, 0)
            # rows are masked as well
            #attention = [batch size, n heads, query len, key len]
        else:
            # no masking necessary
            attention = torch.softmax(energy, dim = -1)
            #attention = [batch size, n heads, query len, key len]
        
        # calculate self-attended output
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, seq len, head dim]
        
        # reorder x
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, seq len, n heads, head dim]
        
        # put x into same form as input
        x = x.view(batch_size, -1, self.hidden_size)
        
        #x = [batch size, seq len, hid dim]
        
        # linear layer - check if necessary
        x = self.z_to_output(x)
        
        #x = [batch size, seq len, hid dim]

        # mask output
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            # mask x for further input
        
        return(x, attention)


class Decoder(nn.Module):
    """Decoder Network
    
    Arguments:
        torch {nn.Module} -- Inherits from nn.Module
    
    Raises:
        NotImplementedError: Only RNNs of type LSTM and GRU are allowed. Otherwise this error this thrown.
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout=0,
            batch_first=True,
            rnn_type='LSTM',
            proba_output=False
        ):
        """Constructor
        
        Arguments:
            input_size {int} -- number of features per time step
            hidden_size {int} -- number of hidden nodes per time step
            num_layers {int} -- number of layers
        
        Keyword Arguments:
            dropout {float} -- percentage of nodes that should switched out at any term (default: {0})
            batch_first {bool} -- if shape is (batch, sequence, features) or not (default: {True})
            rnn_type {str} -- which type of rnn cell should be used (default: {'LSTM'})
            proba_output {bool} -- if it should output a probabilistic distribution (default: {bool})
        
        Raises:
            NotImplementedError: Only RNNs of type LSTM and GRU are allowed. Otherwise this error this thrown.
        """
        super(Decoder, self).__init__()

        # number of features at a given time step
        self.input_size = input_size
        # number of hidden units (at this time step)
        self.hidden_size = hidden_size
        # number of layers
        self.num_layers = num_layers
        # proba
        self.proba_output = proba_output

        # select the rnn_type connection
        if rnn_type == 'LSTM':
            self.model = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers,
                batch_first=batch_first,
                dropout=dropout
            )
        elif rnn_type == 'GRU':
            self.model = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers,
                batch_first=batch_first,
                dropout = dropout
            )
        else:
            raise NotImplementedError

        # define output -- linear adaptor
        self.hidden_to_tanh_mu = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.tanh_mu = nn.Tanh()
        self.tanh_to_mu = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # if you want probalistic ouput measure -- define second output for scale
        if proba_output:
            self.hidden_to_tanh_scale = nn.Linear(self.hidden_size, self.hidden_size)
            self.tanh_scale = nn.Tanh()
            self.than_to_scale = nn.Linear(self.hidden_size, self.hidden_size)

        nn.init.xavier_uniform_(self.hidden_to_tanh_mu.weight)
        nn.init.xavier_uniform_(self.tanh_to_mu.weight)

    def forward(self, x, mask=None):
        """Foreward pass
        
        Arguments:
            x {PackedSequence} -- Batch input, having variational length.
        """
        # Passing in the input and hidden state into the model and obtaining outputs
        h_t, (h_end, c_end) = self.model(x)
        # check if dynamic
        if mask is not None:
            # Reshaping the outputs such that it can be fit into the fully connected layer
            h_t, _ = torch_utils.pad_packed_sequence(h_t, batch_first=True, padding_value=0)
        
        # linear layer
        mu = self.hidden_to_tanh_mu(h_t)
        # tanh
        mu = self.tanh_mu(mu)
        # second linear
        mu = self.tanh_to_mu(mu)
        
        # if probalisitc output
        if self.proba_output:
            scale = self.hidden_to_tanh_scale(h_t)
            scale = self.tanh_scale(h_t)
            scale = self.than_to_scale(h_t)
            scale = scale.exp()
        else:
            scale = torch.zeros(mu.size())
        
        # masking
        if mask is not None:
            mu = mu.masked_fill(mask==0, 0)
            scale = scale.masked_fill(mask==0, 0)
        
        return(h_t, (h_end, c_end), (mu, scale))

