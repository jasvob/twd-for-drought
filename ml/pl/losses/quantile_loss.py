import torch
import torch.nn as nn

from typing import List, Optional

class QuantileLoss(nn.Module):
    """
    Quantile loss function implementation.

    Parameters
    ----------
        quantiles : List[float]
            List of quantiles to compute loss for.
            
        quantile_weights : List[float]
            Weights for aggregating across different quantiles.

        quantile_aggr : str
            Type of aggregation of loss for different quantiles. One of ['mean', 'sum']. Defaults to 'sum'.
    """
    def __init__(self, quantiles : List[float], quantile_weights : Optional[List[float]] = None, quantile_aggr : str = 'sum'):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        if quantile_weights is None:
            self.quantile_weights = torch.ones_like(self.quantiles)
        else:
            self.quantile_weights = torch.tensor(quantile_weights)
        self.quantile_aggr = quantile_aggr
                                    
    def forward(self, preds : torch.Tensor, target : torch.Tensor):
        """
        Computes quantile loss for a given set of quantiles.

        Parameters
        ----------
            preds : torch.Tensor
                Shape (btsz, seq_len, num_quantiles)

            target : torch.Tensor
                Shape (btsz, seq_len, 1)
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        quant = self.quantiles.to(preds.device)

        errors = preds - target
        sq = torch.maximum(-errors, torch.zeros_like(errors))
        s1q = torch.maximum(errors, torch.zeros_like(errors))
        loss = (quant * sq + (1.0 - quant) * s1q)

        loss *= self.quantile_weights.to(preds.device)

        if self.quantile_aggr == 'sum':
            loss = torch.sum(loss, dim=2)
        elif self.quantile_aggr == 'mean':
            loss = torch.mean(loss, dim=2)
        else:
            raise NotImplementedError('Quantile aggregation %s is not implemented.' % self.quantile_aggr)

        return loss.mean()
