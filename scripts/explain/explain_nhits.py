import torch
import torch.nn as nn

import os
import sys
sys.path.append('../')

from typing import Optional

import numpy as np
import pandas as pd

import pytorch_lightning as pl
import pytorch_lightning.cli as pl_cli
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from utils.data.treenet_dataframe import TreeNetDataFrame
from ml.data.treenet_dataset import TreeNetTemporalDataset

from ml.models.nhits import NHitsNet, NHitsBlock, IdentityBasis

import captum
from captum.attr import GradientShap

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Explain N-HITS Predictions',
    description='Explain predictions of N-HITS model using Gradient SHAP method.',
)

arg_parser.add_argument('--config', type=str, help='Path to the configuration file.')   
arg_parser.add_argument('--model_ckpt', type=str, help='Path to the model checkpoint.')   
arg_parser.add_argument('--output_path', type=str, help='Path to where the outputs should be stored.')
arg_parser.add_argument('--series_id', type=str, help='One of [train, val, test].')

if __name__ == "__main__":
    args = arg_parser.parse_args()

    # Create output folders if they don't exist
    explain_path = Path(args.output_path).joinpath('explain')
    explain_path.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)

    cfg_model = cfg['model']['init_args']['model']['init_args']
    cfg_test_dataset = cfg['data']['init_args']['test_dataset']['init_args']
    cfg_train_dataset = cfg['data']['init_args']['train_dataset']['init_args']

    dataset_train = TreeNetTemporalDataset(
        treenet_data_frame=TreeNetDataFrame(
            data_file_path=cfg_train_dataset['treenet_data_frame']['init_args']['data_file_path'],
            metadata_file_path=cfg_train_dataset['treenet_data_frame']['init_args']['metadata_file_path'],
            metadata_filter_fn=cfg_train_dataset['treenet_data_frame']['init_args']['metadata_filter_fn'],
            treedata_filter_fn=cfg_train_dataset['treenet_data_frame']['init_args']['treedata_filter_fn'],
            cache_path=cfg_train_dataset['treenet_data_frame']['init_args']['cache_path'],
            data_filter_fn='ts.dt.year <= 2020 and series_id == %i' % args.series_id,
            min_trees_per_site=cfg_train_dataset['treenet_data_frame']['init_args']['min_trees_per_site'],
            equal_sequence_per_site=cfg_train_dataset['treenet_data_frame']['init_args']['equal_sequence_per_site'],
            interpolate_data=cfg_train_dataset['treenet_data_frame']['init_args']['interpolate_data'],
            drop_na_for_variables=cfg_train_dataset['treenet_data_frame']['init_args']['drop_na_for_variables'],
            drop_na_for_metadata=cfg_train_dataset['treenet_data_frame']['init_args']['drop_na_for_metadata'],
            resample_data=cfg_train_dataset['treenet_data_frame']['init_args']['resample_data'],
            filter_growth_period=cfg_train_dataset['treenet_data_frame']['init_args']['filter_growth_period'],
            filter_by_temperature=cfg_train_dataset['treenet_data_frame']['init_args']['filter_by_temperature'],
            filter_by_frost=cfg_train_dataset['treenet_data_frame']['init_args']['filter_by_frost'],
            filter_min_growth=cfg_train_dataset['treenet_data_frame']['init_args']['filter_min_growth'],
            filter_min_twd=cfg_train_dataset['treenet_data_frame']['init_args']['filter_min_twd'],
            merge_by_site=cfg_train_dataset['treenet_data_frame']['init_args']['merge_by_site'],
        ),
        input_feats=cfg_train_dataset['input_feats'],
        exog_feats=cfg_train_dataset['exog_feats'],
        target_feats=cfg_train_dataset['target_feats'],
        seq_len=cfg_train_dataset['seq_len'],
        max_dist_timesteps=cfg_train_dataset['max_dist_timesteps'],
        normalization=cfg_train_dataset['normalization'],
        norm_inputs=cfg_train_dataset['norm_inputs'],
        norm_exog=cfg_train_dataset['norm_exog'],
        norm_targets=cfg_train_dataset['norm_targets'],
        target_only_growth=cfg_train_dataset['target_only_growth'],
        return_timestamp=True
    )

    dataset_test = TreeNetTemporalDataset(
        treenet_data_frame=TreeNetDataFrame(
            data_file_path=cfg_test_dataset['treenet_data_frame']['init_args']['data_file_path'],
            metadata_file_path=cfg_test_dataset['treenet_data_frame']['init_args']['metadata_file_path'],
            metadata_filter_fn=cfg_test_dataset['treenet_data_frame']['init_args']['metadata_filter_fn'],
            treedata_filter_fn=cfg_test_dataset['treenet_data_frame']['init_args']['treedata_filter_fn'],
            cache_path=cfg_test_dataset['treenet_data_frame']['init_args']['cache_path'],
            data_filter_fn='ts.dt.year > 2021 and series_id == %i' % args.series_id,
            min_trees_per_site=cfg_test_dataset['treenet_data_frame']['init_args']['min_trees_per_site'],
            equal_sequence_per_site=cfg_test_dataset['treenet_data_frame']['init_args']['equal_sequence_per_site'],
            interpolate_data=cfg_test_dataset['treenet_data_frame']['init_args']['interpolate_data'],
            drop_na_for_variables=cfg_test_dataset['treenet_data_frame']['init_args']['drop_na_for_variables'],
            drop_na_for_metadata=cfg_test_dataset['treenet_data_frame']['init_args']['drop_na_for_metadata'],
            resample_data=cfg_test_dataset['treenet_data_frame']['init_args']['resample_data'],
            filter_growth_period=cfg_test_dataset['treenet_data_frame']['init_args']['filter_growth_period'],
            filter_by_temperature=cfg_test_dataset['treenet_data_frame']['init_args']['filter_by_temperature'],
            filter_by_frost=cfg_test_dataset['treenet_data_frame']['init_args']['filter_by_frost'],
            filter_min_growth=cfg_test_dataset['treenet_data_frame']['init_args']['filter_min_growth'],
            filter_min_twd=cfg_test_dataset['treenet_data_frame']['init_args']['filter_min_twd'],
            merge_by_site=cfg_test_dataset['treenet_data_frame']['init_args']['merge_by_site'],
        ),
        input_feats=cfg_test_dataset['input_feats'],
        exog_feats=cfg_test_dataset['exog_feats'],
        target_feats=cfg_test_dataset['target_feats'],
        seq_len=cfg_test_dataset['seq_len'],
        max_dist_timesteps=cfg_test_dataset['max_dist_timesteps'],
        normalization=cfg_test_dataset['normalization'],
        norm_inputs=cfg_test_dataset['norm_inputs'],
        norm_exog=cfg_test_dataset['norm_exog'],
        norm_targets=cfg_test_dataset['norm_targets'],
        target_only_growth=cfg_train_dataset['target_only_growth'],
        return_timestamp=True
    )

    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # We'd like the model to use GPU acceleration, on most systems, this will require move it to the device 'cuda', on MAC this means moving it to device 'mps'
    device = 'cuda'

    blocks = []
    for block in cfg_model['blocks']:
        cfg_block = block['init_args']

        if 'IdentityBasis' in cfg_block['basis_fn']['class_path']:
            basis_fn_instance = IdentityBasis(
                backcast_len=cfg_block['basis_fn']['init_args']['backcast_len'],
                forecast_len=cfg_block['basis_fn']['init_args']['forecast_len'],
                output_dim=cfg_block['basis_fn']['init_args']['output_dim'],
                interpolation_mode=cfg_block['basis_fn']['init_args']['interpolation_mode']
            )
        else:
            basis_fn_instance = None

        if 'MaxPool1d' in cfg_block['pooling']['class_path']:
            pooling_layer = torch.nn.MaxPool1d(
                kernel_size=cfg_block['pooling']['init_args']['kernel_size'],
                stride=cfg_block['pooling']['init_args']['stride'],
                ceil_mode=cfg_block['pooling']['init_args']['ceil_mode']
            )
        elif 'AvgPool1d' in cfg_block['pooling']['class_path']:
            pooling_layer = torch.nn.AvgPool1d(
                kernel_size=cfg_block['pooling']['init_args']['kernel_size'],
                stride=cfg_block['pooling']['init_args']['stride'],
                ceil_mode=cfg_block['pooling']['init_args']['ceil_mode']
            )
        else:
            pooling_layer = None

        if 'activation' in cfg_block.keys():
            if 'ReLU' in cfg_block['activation']['class_path']:
                activation_fn = torch.nn.ReLU()
            else:
                activation_fn = torch.nn.Identity()
        else:
            activation_fn = torch.nn.Identity()

        if 'batch_norm' in cfg_block.keys():
            batch_norm = cfg_block['batch_norm']
        else:
            batch_norm = False

        if 'dropout_prob' in cfg_block.keys():
            dropout_prob = cfg_block['dropout_prob']
        else:
            dropout_prob = 0.25

        block_instance = NHitsBlock(
            backcast_len=cfg_block['backcast_len'],
            forecast_len=cfg_block['forecast_len'],
            inp_time_hist_dim=cfg_block['inp_time_hist_dim'],
            inp_time_futr_dim=cfg_block['inp_time_futr_dim'],
            hidden_dim=cfg_block['hidden_dim'],
            output_dim=cfg_block['output_dim'],
            inp_stat_dim=cfg_block['inp_stat_dim'],
            hidden_stat_dim=cfg_block['hidden_stat_dim'],
            inp_stat_cats=cfg_block['inp_stat_cats'],
            hidden_stat_cats=cfg_block['hidden_stat_cats'],
            num_layers=cfg_block['num_layers'],
            thetas_num_hidden=list(cfg_block['thetas_num_hidden']),
            batch_norm=batch_norm,
            dropout_prob=dropout_prob,
            activation=activation_fn,
            pooling=pooling_layer,
            basis_fn=basis_fn_instance
        )

        blocks.append(block_instance)

    if 'output_activation' in cfg_model.keys():
        if 'ReLU' in cfg_model['output_activation']['class_path']:
            output_activation = torch.nn.ReLU()
        else:
            output_activation = torch.nn.Identity()
    else:
        output_activation = torch.nn.Identity()

    model = NHitsNet(
        naive_level=cfg_model['naive_level'],
        output_activation=output_activation,
        blocks=torch.nn.ModuleList(blocks)
    ).to(device)

    data = torch.load(args.model_ckpt, map_location=torch.device('cpu'))                 # Load Pytorch Lightning checkpoint file
    state_dict = {k.partition('model.')[2]: v for k,v in data['state_dict'].items()}    # Extract state_dict and remove 'model' prefix from weights added by Pytorch Lightning
    model.load_state_dict(state_dict)                                                   # Load state_dict into the model
    model.eval()

    class ModelWrapper(nn.Module):
        def __init__(self, model : NHitsNet) -> None:
            super(ModelWrapper, self).__init__()

            self.model = model

        def forward(self, tgt_bcast : torch.Tensor, inp_bcast : torch.Tensor, inp_fcast : torch.Tensor, hor_idx : Optional[int] = None) -> torch.Tensor:
            preds, _, _, _ = self.model(tgt_bcast, inp_bcast, inp_fcast, None, None)
            preds = preds.permute(0,2,1)
            if hor_idx is not None:
                preds = preds[:, hor_idx, preds.shape[-1] // 2]
            else:
                preds = preds[:, :, preds.shape[-1] // 2]

            return preds
        
    model_wrapper = ModelWrapper(
        model=model,
    ).to(device)
    model_wrapper.eval()

    norm_inputs_gpu = [dataset_test.norm_inputs[0].to('cuda'), dataset_test.norm_inputs[1].to('cuda')]
    norm_exog_gpu = [dataset_test.norm_exog[0].to('cuda'), dataset_test.norm_exog[1].to('cuda')]
    norm_targets_gpu = [dataset_test.norm_targets[0].to('cuda'), dataset_test.norm_targets[1].to('cuda')]

    backcast_len = cfg['model']['init_args']['backcast_len']
    forecast_len = cfg['model']['init_args']['forecast_len']

    base_inputs, base_exog, base_targets, _, _, _ = next(iter(dataloader_train))
    base_tgt_bcast = base_targets[:, :backcast_len, :]
    # defining model input tensors
    n_baselines = 64
    base_tgt_bcast_all = []
    base_inps_bcast_all = []
    base_inps_fcast_all = []
    for i in range(n_baselines):
        base_inputs, base_exog, base_targets, _, _, _ = next(iter(dataloader_train))
        base_tgt_bcast = base_targets[:, :backcast_len, :]
        # defining model input tensors
        base_tgt_bcast = torch.tensor(base_tgt_bcast, requires_grad=True)
        base_inps_bcast = torch.tensor(base_inputs[:, :backcast_len, :], requires_grad=True)
        base_inps_fcast = torch.tensor(base_inputs[:, -forecast_len:, :], requires_grad=True)

        base_tgt_bcast_all.append(torch.mean(base_tgt_bcast, dim=0, keepdim=True))
        base_inps_bcast_all.append(torch.mean(base_inps_bcast, dim=0, keepdim=True))
        base_inps_fcast_all.append(torch.mean(base_inps_fcast, dim=0, keepdim=True))

    base_tgt_bcast = torch.cat(base_tgt_bcast_all, dim=0).to('cuda')
    base_inps_bcast = torch.cat(base_inps_bcast_all, dim=0).to('cuda')
    base_inps_fcast = torch.cat(base_inps_fcast_all, dim=0).to('cuda')

    # defining and applying integrated gradients on ToyModel and the
    dlshap = GradientShap(model_wrapper, multiply_by_inputs=True)

    series_vars = []
    series_attributions = []
    series_preds = []
    series_tstamps = []
    for eval_sample_idx in range(len(dataset_test)):
        print('Computing sample %i / %i ' % (eval_sample_idx+1, len(dataset_test)))
        inputs, exog, targets, site_id, series_id, tstamp = dataset_test[eval_sample_idx]

        inputs = inputs.unsqueeze_(0)
        targets = targets.unsqueeze_(0)

        tgt_bcast = targets[:, :backcast_len, :]
        tgt_fcast = targets[:, -forecast_len:, :].to('cuda')

        # defining model input tensors
        tgt_bcast = torch.tensor(tgt_bcast, requires_grad=True).to('cuda')
        inps_bcast = torch.tensor(inputs[:, :backcast_len, :], requires_grad=True).to('cuda')
        inps_fcast = torch.tensor(inputs[:, -forecast_len:, :], requires_grad=True).to('cuda')

        all_attributions = []
        all_vars = []
        tstamps = []
        for tgt_idx in range(forecast_len):
            attributions, approximation_error = dlshap.attribute((tgt_bcast, inps_bcast, inps_fcast),
                                                            baselines=(base_tgt_bcast, base_inps_bcast, base_inps_fcast),
                                                            stdevs=0.025, 
                                                            n_samples=32,
                                                            target=tgt_idx,
                                                            return_convergence_delta=True)
                                                            
            attrs_inps_all = torch.cat([attributions[1], attributions[2]], dim=1)
            inps_all = torch.cat([inps_bcast, inps_fcast], dim=1)

            attrs_tgt_all = torch.cat([attributions[0], torch.zeros_like(tgt_fcast)], dim=1)
            tgts_all = torch.cat([tgt_bcast, tgt_fcast], dim=1)

            attrs_all = torch.cat([attrs_inps_all, attrs_tgt_all], dim=2)
            inps_all = dataset_test.unnormalize_data(inps_all, dataset_test.normalization[0], norm_inputs_gpu)
            tgts_all = dataset_test.unnormalize_data(tgts_all, dataset_test.normalization[2], norm_targets_gpu)
            vars_all = torch.cat([inps_all, tgts_all], dim=2)

            all_attributions.append(attrs_all)
            all_vars.append(vars_all)

        all_attributions = torch.stack(all_attributions, dim=0)
        all_vars = torch.stack(all_vars, dim=0)

        pred = model_wrapper(tgt_bcast, inps_bcast, inps_fcast)
        pred = dataset_test.unnormalize_data(pred, dataset_test.normalization[2], norm_targets_gpu)

        series_attributions.append(all_attributions.detach().cpu().numpy())
        series_vars.append(all_vars.detach().cpu().numpy())
        series_preds.append(pred.detach().cpu().numpy())
        series_tstamps.append(tstamp.detach().cpu().numpy())

    series_attributions = np.stack(series_attributions)
    series_vars = np.stack(series_vars)
    series_preds = np.stack(series_preds)
    series_tstamps = np.stack(series_tstamps)

    if series_attributions.shape[0] > 1000:
        for i in range((series_attributions.shape[0] // 1000) + 1):
            results = {
                'all_attributions': series_attributions[i*1000:(i+1)*1000],
                'all_vars': series_vars[i*1000:(i+1)*1000],
                'backcast_len': backcast_len,
                'forecast_len': forecast_len,
                'input_feats': dataset_test.input_feats,
                'target_feats': dataset_test.target_feats,
                'pred': series_preds[i*1000:(i+1)*1000],
                'tstamp': series_tstamps[i*1000:(i+1)*1000],
            }
            np.savez(explain_path.joinpath('%s_%i_%i.npz' % (args.config.stem, args.series_id, i), results=results))

    print('Done')