import torch
from torch import nn, Tensor
import torch.nn.utils.rnn as torch_utils
from src.models.anomalia.layers import (
    Encoder, Variational, MultiHeadAttention, Decoder
)
from src.models.anomalia.losses import MaskedMSELoss, InvLogProbLaplaceLoss


class SMAVRA(nn.Module):
    """Multihead self-attention variational recurrent autoencoder"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        # attention_size,
        output_size: int,
        num_layers: int,
        n_heads: int = 2,
        dropout: float = 0.1,
        batch_first: bool = True,
        cuda: bool = True,
        reconstruction_loss_function: str = 'MSELoss',
        mode: str = 'dynamic',
        rnn_type: str = 'LSTM',
        use_variational_attention: bool = True,
        use_proba_output: bool = False
    ):
        """[summary]

        Args:
            input_size (int): size of input features per time step
            hidden_size (int): size of hidden layer
            latent_size (int): size of latent vectors
            output_size (int): [description]
            num_layers (int): num layers for encoder/decoder
            n_heads (int, optional): number of attention head.
                Defaults to 2.
            dropout (float, optional): dropout rate. Defaults to 0.1.
            batch_first (bool, optional): wether batch first or not.
                Defaults to True.
            cuda (bool, optional): whether to use cuda or not.
                Defaults to True.
            reconstruction_loss_function (str, optional): loss for
                reconstruction error. Defaults to 'MSELoss'.
            mode (str, optional): statis or dynamic.
                Defaults to 'dynamic'.
            rnn_type (str, optional): type of rnn to use.
                Defaults to 'LSTM'.
            use_variational_attention (bool, optional): whether to use
                variational attention or not.
                    Defaults to True.
            use_proba_output (bool, optional): whether to use probabilistic
                output or not. Defaults to False.
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
        # self.attention_size = attention_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.reconstruction_loss_function = reconstruction_loss_function
        self.mode = mode
        self.rnn_type = rnn_type
        self.use_variational_attention = use_variational_attention
        self.use_proba_output = use_proba_output

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

        # device = 'cuda:0'
        # if not self.use_cuda:
        #     device = 'cpu'

        self.attention = MultiHeadAttention(
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            dropout=self.dropout,
            device='cuda' if self.use_cuda else 'cpu'  # remove in future
        )

        if self.use_variational_attention:
            # ----------------------------------------------------
            # 4.) Variational - self attention
            self.variational_attention = Variational(
                hidden_size=self.hidden_size,
                latent_size=self.hidden_size,  # attention size not supported
                use_identity=True
            )
            # ----------------------------------------------------
        # 5.) Decoder --> todo #self.attention_size??
        self.decoder = Decoder(
            input_size=self.latent_size + self.hidden_size,
            hidden_size=self.output_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            rnn_type=self.rnn_type,
            proba_output=self.use_proba_output
        )

        # if cuda, move module to gpu
        if self.use_cuda:
            self.cuda()

        if self.mode == 'dynamic':
            self.reconstruction_loss_function = 'MaskedMSELoss'

        # set loss function for reconstruction
        if self.reconstruction_loss_function == 'MSELoss':
            self.loss_fn = nn.MSELoss(reduction='sum')
        if self.reconstruction_loss_function == 'MaskedMSELoss':
            self.loss_fn = MaskedMSELoss(reduction='sum')
        elif (
            self.use_proba_output
            or self.reconstruction_loss_function == 'InvLogProbLaplaceLoss'
        ):
            self.loss_fn = InvLogProbLaplaceLoss(reduction='sum')
            self.reconstruction_loss_function = 'InvLogProbLaplaceLoss'
        else:
            NotImplementedError

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None
    ):
        """forward pass

        Args:
            x (Tensor): input tensor
            mask (Tensor): mask

        Returns:
             (Tensor, (Tensor, Tensor), (Tensor, Tensor)):
                h_t, (h_end, c_end), (mu, scale).
        """
        # ----------------------------------------------------
        # endocer
        h_t, latent, _, attention, lengths = self.encode(x, mask)

        # ----------------------------------------------------
        # decoder
        (decoded_mu, decoded_scale) = self.decode(
            h_t, latent, attention, lengths, mask=mask)
        # decoded, lengths = torch_utils.pad_packed_sequence(
        # out, batch_first=True, padding_value=0)

        # return decoded result
        return((decoded_mu, decoded_scale))

    def encode(self, x: Tensor, mask: Tensor = None):
        """forward pass

        Args:
            x (Tensor): input tensor
            mask (Tensor): mask

        Returns:
            Tensor, Tensor, Tensor, Tensor, Tensor:
                h_t, latent, attention_weights, attention, lengths
        """
        # ----------------------------------------------------
        # endocer
        h_t, (_, _), h_end_out = self.encoder(x)

        # ----------------------------------------------------
        # latent space
        latent = self.variational_latent(h_end_out)

        # ----------------------------------------------------
        # attention
        # unpack sequence, and pad it, if mode dynamic
        if self.mode == 'dynamic':
            h_t, lengths = torch_utils.pad_packed_sequence(
                h_t, batch_first=True, padding_value=0)
        else:
            lengths = None

        # get multihead attention
        attention, attention_weights = self.attention(
            query=h_t, key=h_t, value=h_t, mask=mask)

        # ----------------------------------------------------
        # attention - variational
        if self.use_variational_attention:
            attention = self.variational_attention(attention, mask=mask)

        return h_t, latent, attention_weights, attention, lengths

    def decode(
        self,
        h_t: Tensor,
        latent: Tensor,
        attention: Tensor,
        lengths: Tensor,
        mask: Tensor = None
    ):
        """Decode encoded sequence

        Args:
            h_t (Tensor): hidden state at timestep t.
            latent (Tensor): latent variables.
            attention (Tensor): attention.
            lengths (Tensor): lengths of padding.
            mask (Tensor, optional): mask. Defaults to None.

        Returns:
            Tensor, Tensor: mu, scaled
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
            decoder_input = torch_utils.pack_padded_sequence(
                decoder_input, lengths=lengths,
                batch_first=True, enforce_sorted=False)
        # feed it through decoder
        _, (_, _), (decoded_mu, decoded_scale) = self.decoder(
            decoder_input, mask=mask)
        # decoded, lengths = torch_utils.pad_packed_sequence(
        # out, batch_first=True, padding_value=0)

        # return decoded result
        return((decoded_mu, decoded_scale))

    def kl_loss_latent(self):
        """KL Loss for latent vector

        Returns:
            Tensor: kl div loss of latent
        """
        mu, log_var = (
            self.variational_latent.mu,
            self.variational_latent.logvar
        )

        return(self.kl_div(mu, log_var))

    def kl_loss_attention(
        self,
        mask: Tensor = None
    ):
        """KL Loss for attention vector

        Args:
            mask (Tensor, optional): mask.

        Returns:
            Tensor: kl div loss of attention
        """
        mu, log_var = (
            self.variational_attention.mu,
            self.variational_attention.logvar
        )

        if mask is not None:
            mu = mu.masked_fill(mask == 0, 0)
            log_var = log_var.masked_fill(mask == 0, 0)

        return(self.kl_div(mu, log_var))

    def kl_div(self, mu: Tensor, log_var: Tensor):
        """KL divergence

        see Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Arguments:
            mu (float): mu of distribution
            log_var (float): logvar of distribution

        Returns:
            Tensor: kl div loss of attention
        """
        kld_element = mu.pow(2).add_(
            log_var.exp()).mul_(-1).add_(1).add_(log_var)
        kld = torch.sum(kld_element).mul_(-0.5)
        return(kld)

    def reconstruction_loss(
        self,
        decoded: Tensor,
        x: Tensor,
        mask: Tensor = None
    ):
        """Reconstruction loss

        Arguments:
            decoded (Tensor): decoded sequence
            x (Tensor): input sequence
            mask (Tensor, optional): mask.

        Returns:
            Tensor: reconstruction loss.
        """
        if self.reconstruction_loss_function == 'MaskedMSELoss':
            loss = self.loss_fn(decoded[0], x, mask)
        elif self.reconstruction_loss_function == 'MSELoss':
            loss = self.loss_fn(decoded[0], x)
        else:
            loss = self.loss_fn(decoded[0], decoded[1], x)
        return(loss)

    def compute_loss(
        self,
        decoded: Tensor,
        x: Tensor,
        mask: Tensor = None
    ):
        """Reconstruction loss

        Arguments:
            decoded (Tensor): decoded sequence
            x (Tensor): input sequence
            mask (Tensor, optional): mask.

        Returns:
            Tensor, Tensor, Tensor: reconstruction, kld_latent, kld_attention.
        """
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, _ = torch_utils.pad_packed_sequence(x, batch_first=True)

        if isinstance(decoded, torch.nn.utils.rnn.PackedSequence):
            decoded, _ = torch_utils.pad_packed_sequence(
                decoded, batch_first=True)

        kld_latent = self.kl_loss_latent()
        kld_attention = self.kl_loss_attention(
            mask=mask) if self.use_variational_attention else 0
        reconstruction = self.reconstruction_loss(decoded, x, mask=mask)

        return(reconstruction, kld_latent, kld_attention)
