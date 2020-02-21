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
    def __init__(self, hidden_size, latent_size):
        """Constructor
        
        Arguments:
            hidden_size {int} -- number of features per time step (output from encoder)
            latent_size {int} -- what size the latent vector should be
        """
        super(Variational, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # define linear adapter from h_end to mean, logvar
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
        super(Decoder, self).__init__()

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

        self.fc_linear = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.fc_linear_out = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.act_tanh = nn.Tanh()

        nn.init.xavier_uniform_(self.fc_linear.weight)

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
        out = self.fc_linear(h_t)
        # softplus
        out = self.act_tanh(out)
        # second linear
        out = self.fc_linear_out(out)
        # masking
        if mask is not None:
            out = out.masked_fill(mask==0, 0)

        return(h_t, (h_end, c_end), out)


class MaskedMSELoss(torch.nn.modules.loss._Loss):
    """Masked MSE Loss module
    
    Arguments:
        torch {torch.nn.modules.loss._Loss} -- inherits from _Loss
    """
    def __init__(self, reduction='mean'):
        """Constructor
        
        Arguments:
            reduction {string} -- how MSE should be reduced
        """
        super(MaskedMSELoss, self).__init__()

        if reduction != 'mean':
            NotImplementedError
        
        self.reduction = reduction
    
    def forward(self, x, target, mask):
        """Foreward pass
        
        Arguments:
            x {torch.tensor} -- input tensor (output from neural network)
            target {torch.tensor} -- target tensor 
            mask {torch.tensor} -- mask tensor
        """
        assert x.shape == target.shape == mask.shape

        squared_error = (torch.flatten(x) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)

        if self.reduction == 'mean':
            result = torch.sum(squared_error) / torch.sum(mask)

        return result


class SMAVRA(nn.Module):
    """Multihead self-attention variational recurrent autoencoder
    
    Arguments:
        nn {torch.nn.Module} -- inherits from troch.nn.Module
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        latent_size,
        attention_size,
        output_size,
        num_layers,
        n_heads=2,
        dropout=0.1,
        batch_first=True,
        cuda=True,
        reconstruction_loss_function = 'MSELoss',
        mode='dynamic',
        rnn_type='LSTM'
        ):
        """Constructor
        
        Arguments:
            input_size {int} -- size of input features per time step
            hidden_size {int} -- size of hidden layer
            latent_size {int} -- size of latent vectors
            num_layers {int} -- num layers for encoder and decoder
        
        Keyword Arguments:
            n_heads {int} -- number of heads in multihead self-attention (default: {2})
            dropout {float} -- dropoutrate for multihead self-attention (default: {0.1})
            batch_first {bool} -- batch first flag, if true [batch, sequence, features] (default: {True})
            cuda {boolean} -- whether to use gpu acceleration or not (default: {'cuda:0'})
            reconstruction_loss_function {string} -- type of loss function (default: {'MSELoss'})
            rnn_type {string} -- which cell type should be used for the encoder and decoder (default: {'LSTM'})
        """
        super(SMAVRA, self).__init__()
        self.use_cuda = cuda

        # check if cuda is available to verify if cuda can be used
        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False
        
        # set input variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.attention_size = attention_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.reconstruction_loss_function = reconstruction_loss_function
        self.mode = mode
        self.rnn_type = rnn_type
        
        # set up layer architecture
        # 1.) encoder
        self.encoder = Encoder(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=self.batch_first,
            rnn_type=self.rnn_type
        )
        # ----------------------------------------------------
        # 2.) variational - latent space
        self.variational_latent = Variational(
            hidden_size=self.hidden_size, 
            latent_size=self.latent_size
        )
        # ----------------------------------------------------
        # 3.) Multihead self-attention

        device = 'cuda:0'
        if not self.use_cuda : 
            device = 'cpu'
        
        self.attention = MultiHeadAttention(
            hidden_size=self.hidden_size, 
            n_heads=self.n_heads, 
            dropout=self.dropout, 
            device=device # remove in future
        )
        # ----------------------------------------------------
        # 4.) Variational - self attention
        #self.variational_attention = Variational(
        #    hidden_size=self.hidden_size, 
        #    latent_size=self.attention_size
        #)
        # ----------------------------------------------------
        # 5.) Decoder --> todo
        self.decoder = Decoder(
            input_size=self.latent_size + self.hidden_size, #self.attention_size, 
            hidden_size=self.output_size, 
            num_layers=self.num_layers, 
            batch_first=self.batch_first,
            rnn_type=self.rnn_type
        )

        # if cuda, move module to gpu
        if self.use_cuda:
            self.cuda()
        
        # set loss function for reconstruction
        if self.reconstruction_loss_function == 'MSELoss':
            if self.mode == 'dynamic':
                self.loss_fn = MaskedMSELoss(reduction='sum')
            else:
                self.loss_fn = nn.MSELoss(reduction='sum')
        else:
            NotImplementedError
        
    def forward(self, X, mask=None):
        """Forward pass
        
        Arguments:
            input {tensor} -- input sequence
        """
        # ----------------------------------------------------
        # endocer
        h_t, (_, _), h_end_out = self.encoder(X)

        # ----------------------------------------------------
        # latent space
        latent = self.variational_latent(h_end_out)

        # ----------------------------------------------------
        # attention 
        # unpack sequence, and pad it, if mode dynamic
        if self.mode == 'dynamic':
            h_t, lengths = torch_utils.pad_packed_sequence(h_t, batch_first=True, padding_value=0)

        # get multihead attention
        attention, _ = self.attention(query=h_t, key=h_t, value=h_t, mask=mask)
        
        # ----------------------------------------------------
        # attention - variational
        #attention = self.variational_attention(attention_out, mask=mask)
        
        # ----------------------------------------------------
        # decoder
        # reformat latent to cat it on cols for decoder
        latent = latent.unsqueeze(1)
        # get sequnce length
        seq_len = h_t.shape[1]
        # repeat latent for seq_len
        latent = latent.repeat(1, seq_len, 1)
        # mask latent
        if mask is not None:
            latent = latent.masked_fill(mask == 0, 0)
        # cat it on cols
        decoder_input = torch.cat((attention, latent), 2)
        # pack sequence again
        if self.mode == 'dynamic':
            decoder_input = torch_utils.pack_padded_sequence(decoder_input, lengths=lengths, batch_first=True, enforce_sorted=False)
        # feed it through decoder
        h_t, (_, _), decoded = self.decoder(decoder_input, mask=mask)
        #decoded, lengths = torch_utils.pad_packed_sequence(out, batch_first=True, padding_value=0)
        
        # return decoded result
        return(decoded)

    def encode(self, X, mask=None):
        """Encode Sequence
        
        Arguments:
            input {tensor} -- input sequence
        """
        # ----------------------------------------------------
        # endocer
        h_t, (_, _), h_end_out = self.encoder(X)

        # ----------------------------------------------------
        # latent space
        latent = self.variational_latent(h_end_out)

        # ----------------------------------------------------
        # attention 
        # unpack sequence, and pad it, if mode dynamic
        if self.mode == 'dynamic':
            h_t, lengths = torch_utils.pad_packed_sequence(h_t, batch_first=True, padding_value=0)
        else:
            lengths = None

        # get multihead attention
        attention, attention_weights = self.attention(query=h_t, key=h_t, value=h_t, mask=mask)

        return h_t, latent, attention_weights, attention, lengths
    
    def decode(self, h_t, latent, attention, lengths, mask=None):
        """Forward pass
        
        Arguments:
            input {tensor} -- input sequence
        """

        # ----------------------------------------------------
        # decoder
        # reformat latent to cat it on cols for decoder
        latent = latent.unsqueeze(1)
        # get sequnce length
        seq_len = h_t.shape[1]
        # repeat latent for seq_len
        latent = latent.repeat(1, seq_len, 1)
        # mask latent
        if mask is not None:
            latent = latent.masked_fill(mask == 0, 0)
        # cat it on cols
        decoder_input = torch.cat((attention, latent), 2)
        # pack sequence again
        if self.mode == 'dynamic':
            decoder_input = torch_utils.pack_padded_sequence(decoder_input, lengths=lengths, batch_first=True, enforce_sorted=False)
        # feed it through decoder
        h_t, (_, _), decoded = self.decoder(decoder_input, mask=mask)
        #decoded, lengths = torch_utils.pad_packed_sequence(out, batch_first=True, padding_value=0)
        
        # return decoded result
        return(decoded)  


    def kl_loss_latent(self):
        """KL Loss for latent vector
        """
        mu, log_var = self.variational_latent.mu, self.variational_latent.logvar

        return(self.kl_div(mu, log_var))
    
    def kl_loss_attention(self, mask=None):
        """KL Loss for attention vector
        """
        mu, log_var = self.variational_attention.mu, self.variational_attention.logvar

        if mask is not None:
            mu = mu.masked_fill(mask == 0, 0)
            log_var = log_var.masked_fill(mask == 0, 0)
        
        return(self.kl_div(mu, log_var))

    def kl_div(self, mu, log_var):
        """KL divergence

        see Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        Arguments:
            mu {float} -- mu of distribution
            log_var {float} -- logvar of distribution
        """
        kld_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        kld = torch.sum(kld_element).mul_(-0.5)
        return(kld)
    
    def reconstruction_loss(self, x, decoded, mask=None):
        """Reconstruction loss
        
        Arguments:
            x {tensor} -- input sequence
            decoded {tensor} -- decoded sequence
        """
        if self.mode == 'dynamic':
            loss = self.loss_fn(decoded, x, mask)
        else:
            loss = self.loss_fn(decoded, x)
        return(loss)

    def compute_loss(self, x, decoded, mask=None):
        """Reconstruction loss
        
        Arguments:
            x {tensor} -- input sequence
            decoded {tensor} -- decoded sequence
        """
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, _ = torch_utils.pad_packed_sequence(x, batch_first=True)

        if isinstance(decoded, torch.nn.utils.rnn.PackedSequence):
            decoded, _ = torch_utils.pad_packed_sequence(decoded, batch_first=True)
        
        kld_latent = self.kl_loss_latent()
        kld_attention = 0 #self.kl_loss_attention(mask=mask)
        reconstruction = self.reconstruction_loss(x, decoded, mask=mask)

        return(reconstruction, kld_latent, kld_attention)



#X = [
#    torch.FloatTensor(
#        [
#            [1,1],
#            [2,2],
#            [3,3],
#            [4,4],
#            [5,5],
#            [6,6]
#        ]
#    ),
#    torch.FloatTensor(
#        [
#            [4,4],
#            [5,5],
#            [6,6],
#            [7,7],
#            [8,8],
#            [9,9]
#        ]
#    )
#]
#
#X = torch.stack(X)
##Y = torch.FloatTensor([[4,5],[9,10]]).cuda()
##X = torch.stack(X)
## ####################################################
## variables __init__()
##lengths = [_.size(0) for _ in X]
##X_cat = torch.cat(X, 0)
##X_mean = X_cat.mean(0)
##X_std = X_cat.std(0)
##
##
##X = torch_utils.pad_sequence(sequences=X, batch_first=True)
###X = (X - X_mean) / X_std
##X = torch_utils.pack_padded_sequence(X, lengths=lengths, batch_first=True, enforce_sorted=False)
##X_padded, lengths = torch_utils.pad_packed_sequence(X, batch_first=True, padding_value=0)
## create mask
#mask = None #torch.ones(X_padded.shape).masked_fill(X_padded == 0, 0)
#
#smavra = SMAVRA(
#    input_size=2,
#    hidden_size=8,
#    latent_size=2,
#    attention_size=2,
#    output_size=2,
#    num_layers=2,
#    n_heads=4,
#    dropout=0,
#    batch_first=True,
#    cuda=True,
#    mode='static'
#)
#
#X = X.cuda()
##mask = mask.cuda()
#X_padded = X #X_padded.cuda()
#
## Define hyperparameters
#n_epochs = 400
#lr=0.2
##
### Define Loss, Optimizer
#optimizer = torch.optim.Adam(smavra.parameters(), lr=lr)
#
#smavra.train()
### Training Run
#for epoch in range(1, n_epochs + 1):
#    optimizer.zero_grad() # Clears existing gradients from previous epoch
#    decoded = smavra(X, mask=mask)
#    reconstruction, kld_latent, kld_attention = smavra.compute_loss(X_padded, decoded, mask=mask)
#    loss = (100*reconstruction) + (kld_latent + kld_attention)
#    loss.backward() # Does backpropagation and calculates gradients
#    optimizer.step() # Updates the weights accordingly
#    
#    if epoch%10 == 0:
#        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
#        print(
#            'Recon: {:.4f} \n KLD-Latent: {:.4f} \n KLD-Attention: {:.4f} \n Loss: {:.4f}'.format(
#                reconstruction,
#                kld_latent,
#                kld_attention,
#                loss
#            )
#        )
#
#smavra.eval()
#smavra(X, mask=mask)
#
#((smavra(X, mask=mask) * X_std.cuda()) + X_mean.cuda()).masked_fill(mask==0, 0)
#
#
#smavra.variational_attention.mu

## ####################################################
## init module layers
## ----------------------------------------------------
## 1.) encoder
#encoder = Encoder(
#    input_size=input_size, 
#    hidden_size=hidden_size, 
#    num_layers=num_layers, 
#    batch_first=batch_first
#)
## ----------------------------------------------------
## 2.) variational - latent space
#variational_latent = Variational(
#    hidden_size=hidden_size, 
#    latent_size=latent_size
#)
## ----------------------------------------------------
## 3.) Multihead self-attention
#attention = MultiHeadAttention(
#    hidden_size=hidden_size, 
#    n_heads=n_heads, 
#    dropout=dropout, 
#    device=device
#)
## ----------------------------------------------------
## 4.) Variational - self attention
#variational_attention = Variational(
#    hidden_size=hidden_size, 
#    latent_size=latent_size
#)
## ----------------------------------------------------
## 5.) Decoder --> todo
#decoder = Decoder(
#    input_size=input_size+latent_size, 
#    hidden_size=hidden_size, 
#    num_layers=num_layers, 
#    batch_first=batch_first
#)
## ####################################################
## move everything to gpu
#X = X.cuda()
#encoder.cuda()
#variational_latent.cuda()
#attention.cuda()
#variational_attention.cuda()
#decoder.cuda()
#
## ####################################################
## forward
## ----------------------------------------------------
## endocer
#h_t, (h_end, c_end), h_end_out = encoder(X)
## ----------------------------------------------------
## latent space
#latent = variational_latent(h_end_out)
## ----------------------------------------------------
## attention 
#h_t_padded, _ = torch_utils.pad_packed_sequence(h_t, batch_first=True, padding_value=0)
#mask = torch.ones(h_t_padded.shape).cuda().masked_fill(h_t_padded == 0, 0)
#x, attention_weights = attention(query=h_t_padded, key=h_t_padded, value=h_t_padded, mask=mask)
## ----------------------------------------------------
## attention - variational
#attention = variational_attention(x, mask=mask)
#latent = latent.unsqueeze(1)
#h_size = h_t_padded.shape[1]
#latent = latent.repeat(1, h_size, 1)
#latent = latent.masked_fill(mask == 0, 0)
## ----------------------------------------------------
## decoder
#decoder_input = torch.cat((attention, latent), 2)
#decoder_input = torch_utils.pack_padded_sequence(decoder_input, lengths=lengths, batch_first=True, enforce_sorted=False)
#h_t, (h_end, c_end), h_end_out = decoder(decoder_input)
#decoded, lengths = torch_utils.pad_packed_sequence(h_t, batch_first=True, padding_value=0)
#attention_energy = torch.sum(h_t_padded, dim=2).t()
#
#
#attention_weights = F.softmax(attention_energy, dim=0).unsqueeze(1)
#htt = h_t_padded.transpose(0,1)
#context = attention_weights.bmm(htt)
#context.squeeze(1)
#A
#
#batch_id = 0
#hidden_dim = 1
#attention_weights[0,0,:] @ h_t_padded[0,:,0]
#
#torch.matmul(h_t_padded[0,:,:], attention_weights[0,:,:].transpose(0,1))
# Define hyperparameters
#n_epochs = 400
#lr=0.2
#
## Define Loss, Optimizer
#criterion = nn.MSELoss().cuda()
#optimizer = torch.optim.Adam(demo.parameters(), lr=lr)
##packed_X = packed_X.cuda()
##packed_X = packed_X.cuda()
##_, (h_end, c_end) = demo.rnn(packed_X)
###
##torch_utils.pad_packed_sequence(_, batch_first=True)
## Training Run
#for epoch in range(1, n_epochs + 1):
#    optimizer.zero_grad() # Clears existing gradients from previous epoch
#    packed_X = packed_X.cuda()
#    _, (h_end, c_end), output = demo(packed_X)
#    loss = criterion(output, Y.view(-1).float())
#    loss.backward() # Does backpropagation and calculates gradients
#    optimizer.step() # Updates the weights accordingly
#    
#    if epoch%10 == 0:
#        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
#        print("Loss: {:.4f}".format(loss.item()))
#
#
#
#_, (h_end, c_end), output = demo(packed_X)
#
#output
#
#rnn = nn.LSTM(1, 5, 1)
##h0 = torch.autograd.Variable(torch.zeros(1, 2, 1))
#
#output, (hidden, cell) = rnn(packed_X)
#
#output_unpacked = torch_utils.pad_packed_sequence(output)
#
#output_unpacked[0][:,0,:]
#
#
#
#class Attention(nn.Module):
#    """Attention Layer
#    
#    Arguments:
#        torch {nn.Module} -- Attention from encoder to decoder
#    """
#    def __init__(self, method='dot'):
#        """Constructor
#        
#        Keyword Arguments:
#            method {str} -- which method should be used (default: {'dot'})
#        """
#        super(Attention, self).__init__()
#        self.method = method
#
#    def dot_method(self, hidden):
#        """Dot method
#        
#        Arguments:
#            hidden {[type]} -- input of hidden states (batch, t, features)
#        """
#        return(torch.sum(hidden, dim=2))
#
#    def forward(self, hidden):
#        """[summary]
#        
#        Arguments:
#            hidden {[type]} -- [description]
#        """
#        h_t_padded, _ = torch_utils.pad_packed_sequence(hidden, batch_first=True, padding_value=-float(99999))
#        
#        if self.method == 'dot':
#            attention_energy = self.dot_method(hidden=h_t_padded)
#        else:
#            NotImplementedError
#        
#        attention_energy = attention_energy.t()
#        
#        attention_weights = F.softmax(attention_energy, dim=0).unsqueeze(1)
#        context = attention_weights.bmm(h_t_padded)
#
#        return(attention_weights, context)
#
#
#
#
#
#import torch
#import torch.nn as nn
#from torch.distributions import Normal
#
#class Proba(nn.Module):
#    def __init__(self, size):
#        super(Proba, self).__init__()
#        self.input_to_mu = nn.Linear(size, size)
#        
#    def forward(self, obs):
#        self.mu = self.input_to_mu(obs)
#        self.dist = Normal(self.mu, 0.2)
#        log_prob = self.dist.log_prob(obs)
#        return log_prob.mean()
#
#proba = Proba(10)
#optimizer = torch.optim.Adam(proba.parameters(), lr=1e-3)
#
## fixed observation to reconstruct:
#obs = torch.rand(10) + 10
#
## want to optimize the probability to reconstruct obs:
#if __name__=="__main__":
#
#    for i in range(500):
#        loss = -proba(obs)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        if i%100==0:
#            # should never be larger than 0:
#            print("log_prob of obs =", -loss.item())
#
#
#import torch.nn.functional as F
#F.kl_div(proba.dist.sample().log(), obs)
#
#proba.dist.sample()


#mu = torch.randn((6,2))
#log_var = torch.randn((6,2))
#
#kld_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
#kld_no_mask = torch.sum(kld_element).mul_(-0.5)
#
#null_tensor = torch.zeros((2,2)).float()
#mu = torch.cat([mu, null_tensor], 0)
#log_var = torch.cat([log_var, null_tensor], 0)
#
#kld_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
#kld_mask = torch.sum(kld_element).mul_(-0.5)
#