import torch as torch
from torch import nn as nn
import torch.nn.utils.rnn as torch_utils
import torch.nn.functional as F
from anomalia.layers import Encoder, Variational, MultiHeadAttention, Decoder, MaskedMSELoss


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
        #attention_size,
        output_size,
        num_layers,
        n_heads=2,
        dropout=0.1,
        batch_first=True,
        cuda=True,
        reconstruction_loss_function = 'MSELoss',
        mode='dynamic',
        rnn_type='LSTM',
        use_variational_attention=True
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
        #self.attention_size = attention_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.reconstruction_loss_function = reconstruction_loss_function
        self.mode = mode
        self.rnn_type = rnn_type
        self.use_variational_attention = use_variational_attention
        
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

        if self.use_variational_attention:
            # ----------------------------------------------------
            # 4.) Variational - self attention
            self.variational_attention = Variational(
               hidden_size=self.hidden_size, 
               latent_size=self.hidden_size, # attention size not supported
               use_identity=True
            )
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
        if self.use_variational_attention:
            attention = self.variational_attention(attention, mask=mask)
        
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

        # ----------------------------------------------------
        # attention - variational
        if self.use_variational_attention:
            attention = self.variational_attention(attention, mask=mask)

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
        kld_attention = self.kl_loss_attention(mask=mask) if self.use_variational_attention else 0
        reconstruction = self.reconstruction_loss(x, decoded, mask=mask)

        return(reconstruction, kld_latent, kld_attention)


