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

from ml.models.tft.model import TemporalFusionTransformer

import time

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Evaluate TFT',
    description='Evaluate TFT on the test set and save results to file.',
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

    model = TemporalFusionTransformer(
        num_time_variables=cfg_model['num_time_variables'],
        num_time_categorical_variables=cfg_model['num_time_categorical_variables'],
        num_static_variables=cfg_model['num_static_variables'],
        num_static_categorical_variables=cfg_model['num_static_categorical_variables'],
        time_category_counts=cfg_model['time_category_counts'],
        stat_category_counts=cfg_model['stat_category_counts'],
        hist_variable_idxs=cfg_model['hist_variable_idxs'],
        futr_variable_idxs=cfg_model['futr_variable_idxs'],
        seq_len=cfg_model['seq_len'],
        input_size=cfg_model['input_size'],
        output_size=cfg_model['output_size'],
        hidden_layer_size=cfg_model['hidden_layer_size'],
        dropout_rate=cfg_model['dropout_rate'],
        num_encoder_steps=cfg_model['num_encoder_steps'],
        num_heads=cfg_model['num_heads']
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

            inputs_all = torch.cat([inputs, torch.cat([targets[:, :backcast_len, :], torch.zeros_like(targets[:, backcast_len:, :])], dim=1)], dim=-1)
            start = time.time()
            fcast = model(inputs_all, None, None, None)
            inf_times.append(time.time() - start)
        
            # If the targets were normalized during training, here at test time we have to unnormalize them
            bcast_t = dataset_test.unnormalize_data(targets[:, :backcast_len, :], dataset_test.normalization[2], norm_targets_gpu)
            fcast = dataset_test.unnormalize_data(fcast, dataset_test.normalization[2], norm_targets_gpu)
            fcast_t = dataset_test.unnormalize_data(targets[:, -forecast_len:, :], dataset_test.normalization[2], norm_targets_gpu)
            inputs = dataset_test.unnormalize_data(inputs, dataset_test.normalization[0], norm_inputs_gpu)
            exog = dataset_test.unnormalize_data(exog, dataset_test.normalization[1], norm_exog_gpu)

            inps.append(inputs.detach().cpu())
            exogs.append(exog.detach().cpu())
            bcast_tgts.append(bcast_t.detach().cpu())
            forecasts.append(fcast.detach().cpu())
            fcast_tgts.append(fcast_t.detach().cpu())
            tstamps.append(tstamp.detach().cpu())
            site_ids.append(site_id.detach().cpu())
            series_ids.append(series_id.detach().cpu())

    inps = torch.cat(inps)
    exogs = torch.cat(exogs)
    bcast_tgts = torch.cat(bcast_tgts)
    forecasts = torch.cat(forecasts)
    fcast_tgts = torch.cat(fcast_tgts)
    tstamps = torch.cat(tstamps)
    site_ids = torch.cat(site_ids)
    series_ids = torch.cat(series_ids)
    inf_times = np.array(inf_times)

    results = {
        'inps': inps.numpy(),
        'exogs': exogs.numpy(),
        'bcast_tgts': bcast_tgts.numpy(),
        'fcast_tgts': fcast_tgts.numpy(),
        'forecasts': forecasts.numpy(),
        'tstamps': tstamps.numpy(),
        'site_ids': site_ids.numpy(),
        'series_ids': series_ids.numpy(),
        'inf_times': inf_times
    }

    np.savez(eval_path.joinpath('%s.npz' % args.config.stem), results=results)

    print('Done')