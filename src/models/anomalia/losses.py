import torch


class MaskedMSELoss(torch.nn.modules.loss._Loss):
    """Masked MSE Loss module

    Arguments:
        torch {torch.nn.modules.loss._Loss} -- inherits from _Loss
    """

    def __init__(self, reduction: str = 'mean'):
        """Constructor

        Arguments:
            reduction (string, optional) -- how MSE should be reduced. Defaults to 'mean'.
        """
        super(MaskedMSELoss, self).__init__()

        if reduction != 'mean':
            NotImplementedError

        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Foreward pass

        Args:
            x (torch.Tensor): input tensor (output from neural network)
            target (torch.Tensor): target tensor 
            mask (torch.Tensor): mask tensor

        Returns:
            (torch.Tensor): MSE Loss
        """
        assert x.shape == target.shape == mask.shape

        squared_error = (torch.flatten(x) - torch.flatten(target)
                         ) ** 2.0 * torch.flatten(mask)

        if self.reduction == 'mean':
            result = torch.sum(squared_error) / torch.sum(mask)

        return result


class InvLogProbLaplaceLoss(torch.nn.modules.loss._Loss):
    """Inverse of log probability of laplace distribution"""

    def __init__(self, reduction):
        super(InvLogProbLaplaceLoss, self).__init__()

        if not reduction in ['sum', "mean"]:
            NotImplementedError(
                "InvLogProbLaplaceLoss only supports sum and mean for the reduction parameter")

        self.reduction = reduction

    def forward(self, mu, scale, value):
        """forward pass

        Args:
            mu (torch.Tensor): mu tensor
            scale (torch.Tensor): scale tensor
            value (torch.Tensor): values

        Returns:
            (torch.Tensor): InvLogProbLaplaceLoss

        """
        log_prob = (-torch.log(2 * scale) - torch.abs(value - mu) / scale)

        if self.reduction == 'sum':
            l = log_prob.sum()
        else:
            l = log_prob.mean()

        return(-l)
