import torch

import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import pytorch_lightning as pl
import pytorch_lightning.cli as pl_cli
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from utils.data.treenet_dataframe import TreeNetDataFrame
from ml.data.treenet_dataset import TreeNetTemporalDataset

from ml.models.nhits.model import NHitsNet, NHitsBlock, IdentityBasis

import time

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Evaluate N-HITS',
    description='Evaluate N-HITS on the test set and save results to file.',
)

arg_parser.add_argument('--config', type=str, help='Path to the configuration file.')   
arg_parser.add_argument('--model_ckpt', type=str, help='Path to the model checkpoint.')   
arg_parser.add_argument('--output_path', type=str, help='Path to where the outputs should be stored.')

if __name__ == "__main__":
    args = arg_parser.parse_args()

    # Create output folders if they don't exist
    eval_path = Path(args.output_path).joinpath('eval')
    eval_path.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)

    cfg_model = cfg['model']['init_args']['model']['init_args']
    cfg_test_dataset = cfg['data']['init_args']['test_dataset']['init_args']

    dataset_test = TreeNetTemporalDataset(
        treenet_data_frame=TreeNetDataFrame(
            data_file_path=cfg_test_dataset['treenet_data_frame']['init_args']['data_file_path'],
            metadata_file_path=cfg_test_dataset['treenet_data_frame']['init_args']['metadata_file_path'],
            metadata_filter_fn=cfg_test_dataset['treenet_data_frame']['init_args']['metadata_filter_fn'],
            treedata_filter_fn=cfg_test_dataset['treenet_data_frame']['init_args']['treedata_filter_fn'],
            cache_path=cfg_test_dataset['treenet_data_frame']['init_args']['cache_path'],
            data_filter_fn=cfg_test_dataset['treenet_data_frame']['init_args']['data_filter_fn'],
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
        target_only_growth=cfg_test_dataset['target_only_growth'],
        return_timestamp=True
    )

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
            elif 'GELU' in cfg_block['activation']['class_path']:
                activation_fn = torch.nn.GELU()
            elif 'SiLU' in cfg_block['activation']['class_path']:
                activation_fn = torch.nn.SiLU()
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

    print('Num trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Num params: ', sum(p.numel() for p in model.parameters()))

    data = torch.load(args.model_ckpt, map_location=torch.device('cpu'))                 # Load Pytorch Lightning checkpoint file
    state_dict = {k.partition('model.')[2]: v for k,v in data['state_dict'].items()}    # Extract state_dict and remove 'model' prefix from weights added by Pytorch Lightning
    model.load_state_dict(state_dict)                                                   # Load state_dict into the model
    model.eval()

    norm_inputs_gpu = [dataset_test.norm_inputs[0].to('cuda'), dataset_test.norm_inputs[1].to('cuda')]
    norm_exog_gpu = [dataset_test.norm_exog[0].to('cuda'), dataset_test.norm_exog[1].to('cuda')]
    norm_targets_gpu = [dataset_test.norm_targets[0].to('cuda'), dataset_test.norm_targets[1].to('cuda')]

    backcast_len = cfg['model']['init_args']['backcast_len']
    forecast_len = cfg['model']['init_args']['forecast_len']

    inps = []
    inps_tnet = []
    exogs = []
    bcast_tgts = []
    forecasts = []
    fcast_tgts = []
    decomps = []
    residuals = []
    decomps_bcast = []
    tstamps = []
    site_ids = []
    series_ids = []
    inf_times = []
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            if i % 100 == 0:
                print(i)
                
            inputs, exog, targets, site_id, series_id, tstamp = data
            
            targets = targets.to('cuda')
            inputs = inputs.to('cuda')
            exog = exog.to('cuda')

            tgt_bcast = targets[:, :backcast_len, :]
            start = time.time()
            fcast, decomp, resid, decomp_back = model(tgt_bcast, inputs[:, :backcast_len, :], inputs[:, -forecast_len:, :], None, None)
            inf_times.append(time.time() - start)
            fcast = fcast.permute(0,2,1)
            decomp = decomp.permute(0,3,2,1)
            resid = resid.permute(0,2,1)
            decomp_back = decomp_back.permute(0,3,2,1)

            # If the targets were normalized during training, here at test time we have to unnormalize them
            bcast_t = dataset_test.unnormalize_data(targets[:, :backcast_len, :], dataset_test.normalization[2], norm_targets_gpu)
            fcast = dataset_test.unnormalize_data(fcast, dataset_test.normalization[2], norm_targets_gpu)
            decomp = dataset_test.unnormalize_data(decomp, dataset_test.normalization[2], norm_targets_gpu)
            resid = dataset_test.unnormalize_data(resid, dataset_test.normalization[2], norm_targets_gpu)
            decomp_back = dataset_test.unnormalize_data(decomp_back, dataset_test.normalization[2], norm_targets_gpu)
            fcast_t = dataset_test.unnormalize_data(targets[:, -forecast_len:, :], dataset_test.normalization[2], norm_targets_gpu)
            inputs = dataset_test.unnormalize_data(inputs, dataset_test.normalization[0], norm_inputs_gpu)
            exog = dataset_test.unnormalize_data(exog, dataset_test.normalization[1], norm_exog_gpu)

            inps.append(inputs.detach().cpu())
            exogs.append(exog.detach().cpu())
            bcast_tgts.append(bcast_t.detach().cpu())
            forecasts.append(fcast.detach().cpu())
            fcast_tgts.append(fcast_t.detach().cpu())
            decomps.append(decomp.detach().cpu())
            residuals.append(resid.detach().cpu())
            decomps_bcast.append(decomp_back.detach().cpu())
            tstamps.append(tstamp.detach().cpu())
            site_ids.append(site_id.detach().cpu())
            series_ids.append(series_id.detach().cpu())

    inps = torch.cat(inps)
    exogs = torch.cat(exogs)
    bcast_tgts = torch.cat(bcast_tgts)
    forecasts = torch.cat(forecasts)
    fcast_tgts = torch.cat(fcast_tgts)
    decomps = torch.cat(decomps)
    residuals = torch.cat(residuals)
    decomps_bcast = torch.cat(decomps_bcast)
    tstamps = torch.cat(tstamps)
    site_ids = torch.cat(site_ids)
    series_ids = torch.cat(series_ids)
    inf_times = np.array(inf_times)

    results = {
        'inps': inps.numpy(),
        'exogs': exogs.numpy(),
        'bcast_tgts': bcast_tgts.numpy(),
        'fcast_tgts': fcast_tgts.numpy(),
        'decomps': decomps.numpy(),
        'residuals': residuals.numpy(),
        'decomps_bcast': decomps_bcast.numpy(),
        'forecasts': forecasts.numpy(),
        'tstamps': tstamps.numpy(),
        'site_ids': site_ids.numpy(),
        'series_ids': series_ids.numpy(),
        'inf_times': inf_times
    }

    np.savez(eval_path.joinpath('%s.npz' % args.config.stem), results=results)

    print('Done')