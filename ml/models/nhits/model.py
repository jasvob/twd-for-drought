import torch
import torch.nn as nn
import numpy as np

from typing import Generic, Tuple, List, Callable, Optional
import time

from ml.models.revin import RevInstanceNorm

class StaticFeatureEncoder(nn.Module):
    """
    Feature encoder for the statis exogenous variables.

    Parameters
    ----------
        in_features : int
            Number of input features.

        out_features : int
            Number of output features.    
    """
    def __init__(self, in_features : int, out_features : int) -> None:
        super(StaticFeatureEncoder, self).__init__()

        self.encoder = nn.Sequential(*[
            nn.Dropout(p=0.5),
            nn.Linear(in_features, out_features),
            nn.ReLU()
        ])
    
    def forward(self, x : torch.Tensor):
        """
        Parameters
        ----------
            x : torch.Tensor
                Static exogenous variables to be encoded.

        Returns
        -------
            : torch.Tensor
                Encoding of the static exogenous variables. 
        """
        x = self.encoder(x)
        return x

         
class NHitsBlock(nn.Module):
    """
    Building block of the NHits model.

    Parameters
    ----------
        backcast_len : int
            Length of the signal history.

        forecast_len : int
            Length of the signal to forecast.

        inp_time_dim : int
            Number of input time-series features.

        inp_stat_dim : int
            Number of input static features.

        hidden_dim : int
            Hidden layer feature size.

        hidden_stat_dim : int
            Hidden layer feature size for static features.

        output_dim : int
            Number of output dimensions.

        basis_fn : nn.Module
            Basis function that defines the functionality of the base block. 
            One of: 
                IdentityBasis

        thetas_num_hidden : List[int]
            Number of hidden units for the computation of thetas.

        batch_norm : bool
            Whether to perform batch normalization or not. Defaults to False.

        dropout_prob : float
            Feature dropout probability. Defaults to 0.25.

        pooling : nn.Module
            Pooling module for subsampling the input signal during the hierarchical interpolation. Defaults to nn.MaxPool1d.

        activation : nn.Module
            Activation function to apply to the layer output. Defaults to nn.Identity(), i.e. no activation.
    """
    def __init__(self, 
                backcast_len : int, forecast_len : int,
                inp_time_hist_dim : int, inp_time_futr_dim : int,
                inp_stat_dim : int,
                hidden_dim : int, hidden_stat_dim : int, 
                output_dim : int,
                num_layers : int,
                basis_fn : nn.Module,
                thetas_num_hidden : List[int],
                batch_norm : bool = False,
                dropout_prob : float = 0.25,
                pooling : nn.Module = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                activation : nn.Module = nn.Identity(),
                inp_stat_cats : Optional[List[int]] = None,
                hidden_stat_cats : Optional[List[int]] = None
                ) -> None:
        super(NHitsBlock, self).__init__()

        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.inp_time_hist_dim = inp_time_hist_dim
        self.inp_time_futr_dim = inp_time_futr_dim
        self.inp_stat_dim = inp_stat_dim
        self.num_layers = num_layers
        self.pooling_layer = pooling
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_stat_dim = hidden_stat_dim
        self.basis_fn = basis_fn
        self.theta_num_hidden = thetas_num_hidden
        self.batch_norm = batch_norm
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.inp_stat_cats = inp_stat_cats
        self.hidden_stat_cats = hidden_stat_cats
        
        # Depending on the type of pooling module, make sure that kernel_size is an integer, not a tuple
        if type(self.pooling_layer.kernel_size) is tuple:
            ker_size = self.pooling_layer.kernel_size[0] 
        else:
            ker_size = self.pooling_layer.kernel_size
        
        self.backcast_pooled_len = int(np.ceil(backcast_len / ker_size))
        self.forecast_pooled_len = int(np.ceil(forecast_len / ker_size))
        self.thetas_dim = self.backcast_pooled_len + self.output_dim * self.forecast_pooled_len
        sum_hidden_stat_cats = np.sum(self.hidden_stat_cats) if self.hidden_stat_cats is not None else 0
        thetas_num_hidden = [self.backcast_pooled_len + self.backcast_pooled_len * inp_time_hist_dim + self.forecast_pooled_len * inp_time_futr_dim + hidden_stat_dim + sum_hidden_stat_cats] + thetas_num_hidden[:-1] + [thetas_num_hidden[-1] * output_dim]

        hidden_layers = []
        for i in range(self.num_layers):
            # Batch norm after activation
            hidden_layers.append(nn.Linear(thetas_num_hidden[i], thetas_num_hidden[i+1]))

            if self.batch_norm:
                hidden_layers.append(nn.LayerNorm(thetas_num_hidden[i+1]))

            hidden_layers.append(self.activation)

            if self.dropout_prob > 0.0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        # Add output layer
        hidden_layers.append(nn.Linear(thetas_num_hidden[-1], self.thetas_dim))
        self.block_mlp = nn.Sequential(*hidden_layers)

        self.layer_norm = nn.LayerNorm(self.thetas_dim)

        # If there are some static exogenous features, encode them
        if self.inp_stat_dim > 0 and self.hidden_stat_dim > 0:
            self.static_encoder = StaticFeatureEncoder(self.inp_stat_dim, self.hidden_stat_dim)
        
        if self.inp_stat_cats is not None and self.hidden_stat_cats is not None:
            self.categorical_encoders = nn.ModuleList()
            for i in range(len(self.inp_stat_cats)):
                self.categorical_encoders.append(nn.Embedding(num_embeddings=self.inp_stat_cats[i], embedding_dim=self.hidden_stat_cats[i], max_norm=1.0))

    def forward(self, backcast_y : torch.Tensor, backcast_x_t : torch.Tensor, forecast_x_t : torch.Tensor, x_stat : torch.Tensor, x_cat : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
            backcast_y : torch.Tensor
                Tensor containing history for target time-series we want to forecast.

            backcast_x_t : torch.Tensor
                Tensor containing history for input time-series.

            forecast_x_t : torch.Tensor
                Tensor containing future for target time-series we want to forecast. 

            x_stat : torch.Tensor
                Exogenous static variables. 

        Returns
        -------
            backcast : torch.Tensor
                Tensor containing backcast for target time-series from this block.

            forecast : torch.Tensor
                Tensor containing forecast for target time-series from this block.
        """

        # Downsample the input using the pooling layer
        backcast_y = self.pooling_layer(backcast_y)

        btsz = backcast_y.shape[0]
        if self.inp_time_hist_dim > 0:
            backcast_x_t = self.pooling_layer(backcast_x_t)
            backcast_y = torch.cat([backcast_y.reshape(btsz, -1), backcast_x_t.reshape(btsz, -1)], dim=1)

        if self.inp_time_futr_dim > 0:
            forecast_x_t = self.pooling_layer(forecast_x_t)
            backcast_y = torch.cat([backcast_y.reshape(btsz, -1), forecast_x_t.reshape(btsz, -1)], dim=1)

        # Static exogenous features
        if self.inp_stat_dim > 0 and self.hidden_stat_dim > 0:
            x_stat = self.static_encoder(x_stat)
            backcast_y = torch.cat([backcast_y, x_stat], dim=1)

        if self.inp_stat_cats is not None and self.hidden_stat_cats is not None:
            x_cat = torch.cat([self.categorical_encoders[i](x_cat[..., i]) for i in range(len(self.categorical_encoders))], dim=-1)
            backcast_y = torch.cat([backcast_y, x_cat], dim=1)
          
        # Compute local projection weights theta
        thetas = self.block_mlp(backcast_y)
        thetas = self.layer_norm(thetas)

        thetas_backcast = thetas[:, None, :self.backcast_pooled_len]
        thetas_forecast = thetas[:, None, self.backcast_pooled_len:].reshape(-1, self.output_dim, self.forecast_pooled_len)

        backcast, forecast = self.basis_fn(thetas_backcast, thetas_forecast)

        return backcast, forecast


class NHitsNet(nn.Module):
    """
    Implementation of  NHits model as described in:
    Challu Ch., etal., 2022: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
    https://arxiv.org/pdf/2104.05522.pdf
    
    Implementation available here:
    https://github.com/cchallu/n-hits

    Parameters
    ----------
        blocks : List[nn.Module] 
            List of blocks that the NHits network is composed of.

    """
    def __init__(self, blocks : List[nn.Module], output_activation : nn.Module = nn.Identity(), naive_level : bool = True) -> None:
        super(NHitsNet, self).__init__()

        self.revin_target = RevInstanceNorm(num_features=1)

        self.naive_level = naive_level

        self.blocks = nn.ModuleList(blocks)
        self.output_activation = output_activation

    def forward(self, backcast_y : torch.Tensor, backcast_x_t : torch.Tensor, forecast_x_t : torch.Tensor, x_stat : torch.Tensor, x_cat : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
            backcast_y : torch.Tensor
                Tensor containing history for target time-series we want to forecast.

            backcast_x_t : torch.Tensor
                Tensor containing history for input time-series.

            forecast_x_t : torch.Tensor
                Tensor containing future for target time-series we want to forecast. 

            x_stat : torch.Tensor
                Exogenous static variables. 

        Returns
        -------
            forecast : torch.Tensor
                Tensor containing forecast for target time-series.

            block_forecasts : torch.Tensor
                List of tensors containing partial forecasts for target time-series from each block in the network.
        """

        # For targets
        backcast_y = self.revin_target(backcast_y, mode='norm')

        residuals = backcast_y.permute(0, 2, 1)
        backcast_x_t = backcast_x_t.permute(0, 2, 1)
        if forecast_x_t is not None:
            forecast_x_t = forecast_x_t.permute(0, 2, 1)

        forecast = backcast_y[:, -1:].repeat(
            1, self.blocks[0].forecast_len, 1
        )
        forecast = forecast.repeat_interleave(
            torch.tensor(self.blocks[0].output_dim, device=forecast.device), dim=2
        ).permute(0,2,1)
        
        if self.naive_level:
            block_forecasts = [forecast] 
            block_backcasts = [backcast_y[:, -1:].repeat(1, self.blocks[0].backcast_len, 1).permute(0, 2, 1)]
            residuals = residuals - block_backcasts[-1]
        else:
            forecast = torch.zeros_like(
                forecast, device=forecast.device
            )
            block_forecasts = []
            block_backcasts = []
    
        for i, block in enumerate(self.blocks):
            block_backcast, block_forecast = block(residuals, backcast_x_t, forecast_x_t, x_stat, x_cat)
            residuals = residuals - block_backcast
            forecast = forecast + block_forecast
            block_forecasts.append(self.revin_target(block_forecast, mode='denorm'))
            block_backcasts.append(self.revin_target(block_backcast, mode='denorm'))

        # (btsz, num_blocks, time_dim)
        block_forecasts = torch.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2, 3)
        block_backcasts = torch.stack(block_backcasts)
        block_backcasts = block_backcasts.permute(1, 0, 2, 3)

        # Normalize block forecasts and final forecast back into original range
        forecast = self.revin_target(forecast, mode='denorm')
        residuals = self.revin_target(residuals, mode='denorm')

        return self.output_activation(forecast), block_forecasts, residuals, block_backcasts

class IdentityBasis(nn.Module):
    """
    Building block of the NHits model.

    Parameters
    ----------
        backcast_len : int
            Length of the signal history.

        forecast_len : int
            Length of the signal to forecast.

        output_dim : int
            Number of output time-series features.

        interpolation_mode : str
            Type of interpolation to use to interpolate the signal.
    """
    def __init__(self, backcast_len : int, forecast_len : int, output_dim : int, interpolation_mode : str) -> None:
        super(IdentityBasis, self).__init__()

        assert (interpolation_mode in ['linear', 'nearest']) or ('cubic' in interpolation_mode)

        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.output_dim = output_dim
        self.interpolation_mode = interpolation_mode

    def forward(self, thetas_backcast : torch.Tensor, thetas_forecast : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
            thetas_backcast : torch.Tensor
                Tensor containing the learned basis functions parameters theta for the history.

            thetas_backcast : torch.Tensor
                Tensor containing the learned basis functions parameters theta for the forecast.

        Returns
        -------
            backcast : torch.Tensor
                Tensor containing backcast for target time-series from this block.

            forecast : torch.Tensor
                Tensor containing forecast for target time-series from this block.
        """
        
        if self.interpolation_mode in ['nearest', 'linear']:
            backcast = torch.nn.functional.interpolate(thetas_backcast, size=self.backcast_len, mode=self.interpolation_mode)
            forecast = torch.nn.functional.interpolate(thetas_forecast, size=self.forecast_len, mode=self.interpolation_mode)
        elif 'cubic' in self.interpolation_mode:
            backcast = torch.nn.functional.interpolate(thetas_backcast[:, :, None, :], size=self.backcast_len, mode='bicubic')
            forecast = torch.nn.functional.interpolate(thetas_forecast[:, :, None, :], size=self.forecast_len, mode='bicubic')
            backcast = backcast[:, :, 0, :]
            forecast = forecast[:, :, 0, :]

        return backcast, forecast




