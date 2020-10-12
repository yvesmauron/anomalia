import torch as torch
from torch import nn as nn
import torch.nn.utils.rnn as torch_utils


class Encoder(nn.Module):
    """Encoder Network"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0,
        batch_first: bool = True,
        rnn_type: str = 'LSTM'
    ):
        """Create Encoder

        Args:
            input_size (int): number of features per time step
            hidden_size (int): number of hidden nodes per time step
            num_layers (int): number of layers
            dropout (float, optional): percentage of nodes that should
                switched out at any term. Defaults to 0.
            batch_first (bool, optional): if shape is (batch, sequence,
                features) or not. Defaults to True.
            rnn_type (str, optional): what rnn type to choose.
                Defaults to 'LSTM'.

        Raises:
            NotImplementedError: if invalid rnn type is selected
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
                dropout=dropout
            )
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: ouput tensor
        """
        # Passing in the input and hidden state into the model and obtaining
        # outputs
        h_t, (h_end, c_end) = self.model(x)
        # Reshaping the outputs such that it can be fit into the fully
        # connected layer
        # h_end.shape[1] -> batch size
        # h_end.shape[2] -> hidden size
        # creates the output that fits
        h_end_out = h_end[-1, :, :]
        # out = self.linear_adapter(lin_input)
        return(h_t, (h_end, c_end), h_end_out)


class Variational(nn.Module):
    """Variation Layer of Variational AutoEncoder"""

    def __init__(self, hidden_size: int, latent_size: int,
                 use_identity: bool = False):
        """Variational

        Args:
            hidden_size (int): number of features per time step
                (output from encoder)
            latent_size (int): what size the latent vector should be
            use_identity (bool, optional): if identity should be used.
                Defaults to False.
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
            self.hidden_to_logvar = nn.Linear(
                self.hidden_size, self.latent_size)

            # Fills the input Tensor with values according to the method
            # described in: "Understanding the difficulty of training deep
            # feedforward neural networks" - Glorot, X. & Bengio, Y. (2010)
            nn.init.xavier_uniform_(self.hidden_to_mu.weight)
            nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, hidden: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass

        Args:
            hidden (torch.Tensor): last hidden state of the Encoder.
            mask (torch.Tensor, optional): mask. Defaults to None.
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
        if next(self.parameters()).is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()

        eps = nn.parameter.Parameter(eps, requires_grad=False)
        # return latent output
        latent = eps.mul(std).add(self.mu)

        # mask output
        if mask is not None:
            latent = latent.masked_fill(mask == 0, 0)

        return(latent)


class MultiHeadAttention(nn.Module):
    """Multihead self-attention"""

    def __init__(self, hidden_size: int, attention_size: int, n_heads: int,
                 dropout: float, device: str):
        """Constructor

        Args:
            hidden_size (int): hidden size of the input.
            attention_size (int): attention size
            n_heads (int): number of heads (attention heads)
            dropout (double): dropout rate
            device (string): cuda or cpu
        """
        super().__init__()

        assert attention_size % n_heads == 0

        # init values
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = attention_size // n_heads
        self.attention_size = attention_size

        # linear layers
        self.hidden_to_query = nn.Linear(hidden_size, attention_size)
        self.hidden_to_key = nn.Linear(hidden_size, attention_size)
        self.hidden_to_value = nn.Linear(hidden_size, attention_size)

        # potentially remove
        self.z_to_output = nn.Linear(attention_size, hidden_size)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # scale
        self.scale = nn.parameter.Parameter(
            torch.sqrt(torch.FloatTensor([self.head_size]))
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """Forward pass

        Args:
            query (torch.Tensor): query tensor
            key (torch.Tensor): key tensor
            value (torch.Tensor): value tensor
            mask (torch.Tensor): optional mask for variable length input
                (default: {None})

        Returns:
            (torch.Tensor, torch.Tensor): (attended output, attention weights)
        """
        # get batch size
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        # transform query, key and value
        Q = self.hidden_to_query(query)
        K = self.hidden_to_key(key)
        V = self.hidden_to_value(value)

        # mask output, to prevent wrong energy calculation
        # if mask is not None:
        #    Q = Q.masked_fill(mask == 0, 0)
        #    K = K.masked_fill(mask == 0, 0)
        #    V = V.masked_fill(mask == 0, 0)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        # reorder dimensions
        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_size).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # calculate energy
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, seq len, seq len]

        # mask energy and attention
        if mask is not None:
            energy_mask = mask.view(
                batch_size,
                -1,
                self.n_heads,
                self.head_size
            ).permute(0, 2, 1, 3)
            # energy_mask = [batch size, n heads, query len, head dim]
            energy_mask = torch.matmul(
                energy_mask, energy_mask.permute(0, 1, 3, 2)) == 0
            # energy_mask = [batch size, n heads, seq len, seq len]
            energy = energy.masked_fill(energy_mask, -1e10)
            # masked input for attention
            attention = torch.softmax(energy, dim=-1)
            # attention -> not masked on rows
            attention = attention.masked_fill(energy_mask, 0)
            # rows are masked as well
            # attention = [batch size, n heads, query len, key len]
        else:
            # no masking necessary
            attention = torch.softmax(energy, dim=-1)
            # attention = [batch size, n heads, query len, key len]

        # calculate self-attended output
        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, seq len, head dim]

        # reorder x
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, seq len, n heads, head dim]

        # put x into same form as input
        x = x.view(batch_size, -1, self.attention_size)

        # x = [batch size, seq len, hid dim]

        # linear layer - check if necessary
        x = self.z_to_output(x)

        # x = [batch size, seq len, hid dim]

        # mask output
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            # mask x for further input

        return(x, attention)


class Decoder(nn.Module):
    """Decoder Network"""

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
        """Create Encoder

        Args:
            input_size (int): number of features per time step
            hidden_size (int): number of hidden nodes per time step
            num_layers (int): number of layers
            dropout (float, optional): percentage of nodes that should
                switched out at any term. Defaults to 0.
            batch_first (bool, optional): if shape is (batch, sequence,
                features) or not. Defaults to True.
            rnn_type (str, optional): what rnn type to choose.
                Defaults to 'LSTM'.
            proba_output (bool): if it should output a probabilistic
                distribution. Defaults to False.

        Raises:
            NotImplementedError: Only RNNs of type LSTM and GRU are allowed.
                Otherwise this error this thrown.
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
                dropout=dropout
            )
        else:
            raise NotImplementedError

        # define output -- linear adaptor
        self.hidden_to_tanh_mu = nn.Linear(
            self.hidden_size, self.hidden_size * 2)
        self.tanh_mu = nn.Tanh()
        self.tanh_to_mu = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # if you want probalistic ouput measure -- define second output for
        # scale
        if proba_output:
            self.hidden_to_tanh_scale = nn.Linear(
                self.hidden_size, self.hidden_size)
            self.tanh_scale = nn.Tanh()
            self.than_to_scale = nn.Linear(self.hidden_size, self.hidden_size)

        nn.init.xavier_uniform_(self.hidden_to_tanh_mu.weight)
        nn.init.xavier_uniform_(self.tanh_to_mu.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """forward pass

        Args:
            x (torch.Tensor): input tensor
            mask (torch.Tensor): mask

        Returns:
             (torch.Tensor, (torch.Tensor, torch.Tensor),
             (torch.Tensor, torch.Tensor)): h_t, (h_end, c_end), (mu, scale).
        """
        # Passing in the input and hidden state into the model and obtaining
        # outputs
        h_t, (h_end, c_end) = self.model(x)
        # check if dynamic
        if mask is not None:
            # Reshaping the outputs such that it can be fit into the fully
            # connected layer
            h_t, _ = torch_utils.pad_packed_sequence(
                h_t, batch_first=True, padding_value=0)

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
            mu = mu.masked_fill(mask == 0, 0)
            scale = scale.masked_fill(mask == 0, 0)

        return(h_t, (h_end, c_end), (mu, scale))
