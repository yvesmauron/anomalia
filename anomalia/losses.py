import torch


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


class InvLogProbLaplaceLoss(torch.nn.modules.loss._Loss):
    """Inverse of log probability of laplace distribution
    
    Arguments:
        torch {torch.nn.modules.loss._Loss} -- inherits from _Loss
    """
    def __init__(self, reduction):
        super(InvLogProbLaplaceLoss, self).__init__()

        if not reduction in ['sum', "mean"]:
            NotImplementedError("InvLogProbLaplaceLoss only supports sum and mean for the reduction parameter")
        

        self.reduction = reduction
    
    def forward(self, mu, scale, value):
        """forward pass
        
        Arguments:
            mu {torch.tensor} -- mu tensor
            scale {torch.tensor} -- scale tensor
            value {torch.tensor} -- values
        """
        log_prob = (-torch.log(2 * scale) - torch.abs(value - mu) / scale)
        
        if self.reduction == 'sum':
            l = log_prob.sum()
        else:
            l = log_prob.mean()
        
        return(-l)
        