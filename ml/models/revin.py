import torch
import torch.nn as nn

class RevInstanceNorm(nn.Module):
    def __init__(self, num_features : int, eps : float = 1e-5, affine : bool = True):
        """
        Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift
        https://openreview.net/forum?id=cGDAkQo1C0p

        Paramters
        ---------
            num_features : int
                Number of feature or channels.

            eps : float
                Value added for numerical stability.

            affine : bool
                If True, RevIN has learnable affine parameters.
        """
        super(RevInstanceNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x : torch.Tensor, mode : str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'norm_only':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x : torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x : torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x : torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x