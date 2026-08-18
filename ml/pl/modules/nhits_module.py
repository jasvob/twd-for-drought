import logging
from typing import Any, Sequence, Tuple, Union, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torchmetrics import MeanAbsoluteError, MeanSquaredError, SymmetricMeanAbsolutePercentageError
from ml.pl.metrics.mase import MeanAbsoluteScaledPrecision
from pytorch_lightning.cli import OptimizerCallable, LRSchedulerCallable

class NHitsModule(pl.LightningModule):
    def __init__(self, backcast_len : int, forecast_len : int, model : torch.nn.Module, loss : torch.nn.Module, optimizer : OptimizerCallable = torch.optim.Adam, lr_scheduler : LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR, monitor : str = None, interval : str = 'epoch') -> None:
        super(NHitsModule, self).__init__()
        
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.model = model
        self.loss_fn = loss

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.monitor = monitor
        self.interval = interval

        self.train_acc_mae = MeanAbsoluteError()
        self.train_acc_mse = MeanSquaredError()
        self.train_acc_smape = SymmetricMeanAbsolutePercentageError()
        self.train_acc_mase = MeanAbsoluteScaledPrecision()

        self.val_acc_mae = MeanAbsoluteError()
        self.val_acc_mse = MeanSquaredError()
        self.val_acc_smape = SymmetricMeanAbsolutePercentageError()
        self.val_acc_mase = MeanAbsoluteScaledPrecision()

        self.test_acc_mae = MeanAbsoluteError()
        self.test_acc_mse = MeanSquaredError()
        self.test_acc_smape = SymmetricMeanAbsolutePercentageError()
        self.test_acc_mase = MeanAbsoluteScaledPrecision()

    def forward(self, backcast : torch.Tensor, feats_backcast : torch.Tensor, feats_forecast : torch.Tensor, feats_stat : torch.Tensor, feats_cat : torch.Tensor) -> torch.Tensor:
        return self.model(backcast, feats_backcast, feats_forecast, feats_stat, feats_cat)

    def step(self, batch: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        inputs, _, targets = batch

        #inputs = inputs[:, 0, :, :]
        #targets = targets[:, 0, :, :]

        y_pred, _, _, _ = self(targets[:, :self.backcast_len, :], inputs[:, :self.backcast_len, :], inputs[:, self.backcast_len:, :], None, None)
        y_pred = y_pred.permute(0, 2, 1)
      
        loss = self.loss_fn(y_pred, targets[:, -self.forecast_len:, :])
        
        return y_pred, targets[:, self.backcast_len-1:self.backcast_len, :], loss

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        preds, naive, loss = self.step(batch)   
        preds = preds[..., preds.shape[-1]//2].unsqueeze(-1).contiguous() 
        tgts = batch[-1][:, self.backcast_len:, :].contiguous()  
        
        self.train_acc_mae.update(preds, tgts)
        self.train_acc_mse.update(preds, tgts)
        self.train_acc_smape.update(preds, tgts)
        self.train_acc_mase.update(preds, naive, tgts)

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('train_mae', self.train_acc_mae, on_step=False, on_epoch=True)
        self.log('train_mse', self.train_acc_mse, on_step=False, on_epoch=True)
        self.log('train_smape', self.train_acc_smape, on_step=False, on_epoch=True)
        self.log('train_mase', self.train_acc_mase, on_step=False, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        preds, naive, loss = self.step(batch)    
        preds = preds[..., preds.shape[-1]//2].unsqueeze(-1).contiguous()     
        tgts = batch[-1][:, self.backcast_len:, :].contiguous()  
       
        self.val_acc_mae.update(preds, tgts)
        self.val_acc_mse.update(preds, tgts)
        self.val_acc_smape.update(preds, tgts)
        self.val_acc_mase.update(preds, naive, tgts)

        self.log('val_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('val_mae', self.val_acc_mae, on_step=False, on_epoch=True)
        self.log('val_mse', self.val_acc_mse, on_step=False, on_epoch=True)
        self.log('val_smape', self.val_acc_smape, on_step=False, on_epoch=True)
        self.log('val_mase', self.val_acc_mase, on_step=False, on_epoch=True)

        return {'val_loss': loss}

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        preds, naive, loss = self.step(batch)    
        preds = preds[..., preds.shape[-1]//2].unsqueeze(-1).contiguous()    
        tgts = batch[-1][:, self.backcast_len:, :].contiguous() 
      
        self.test_acc_mae.update(preds, tgts)
        self.test_acc_mse.update(preds, tgts)
        self.test_acc_smape.update(preds, tgts)
        self.test_acc_mase.update(preds, naive, tgts)

        self.log('test_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('test_mae', self.test_acc_mae, on_step=False, on_epoch=True)
        self.log('test_mse', self.test_acc_mse, on_step=False, on_epoch=True)
        self.log('test_smape', self.test_acc_smape, on_step=False, on_epoch=True)
        self.log('test_mase', self.test_acc_mase, on_step=False, on_epoch=True)

    def configure_optimizers(self,) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """

        optimizer = self.optimizer(self.model.parameters())
        lr_scheduler = self.lr_scheduler(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': self.monitor, 'interval': self.interval}}
