from torchmetrics import Metric
import torch

class MeanAbsoluteScaledPrecision(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("naive", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, naive : torch.Tensor, targets: torch.Tensor) -> None:
        self.preds.append(preds)
        self.naive.append(naive.repeat(1, preds.shape[1], 1))
        self.targets.append(targets)

    def compute(self):
        # parse inputs
        preds = torch.cat(self.preds, dim=0)
        naive = torch.cat(self.naive, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        mase = torch.mean(torch.abs(preds - targets) / torch.mean(torch.abs(naive - targets), dim=1, keepdims=True), dim=0) 

        return mase.mean()