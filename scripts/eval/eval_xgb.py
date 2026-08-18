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

import pickle

import time

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Evaluate Gradient Boosting Regressor',
    description='Evaluate gradient boosting regressor on the test set and save results to file.',
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

    # Load the RF model
    with open(args.model_ckpt, 'rb') as f:
        model = pickle.load(f)
    model.verbose = 0
    model.n_jobs = 1

    n_params = sum(tree[0].tree_.node_count for tree in model.estimators_) * 5
    print('# Trainable params: ', n_params)

    norm_inputs = [dataset_test.norm_inputs[0], dataset_test.norm_inputs[1]]
    norm_exog = [dataset_test.norm_exog[0], dataset_test.norm_exog[1]]
    norm_targets = [dataset_test.norm_targets[0], dataset_test.norm_targets[1]]

    backcast_len = cfg['model']['backcast_len']
    forecast_len = cfg['model']['forecast_len']

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
            tgt_fcast = targets[:, -(cfg_test_dataset['seq_len']-backcast_len):, :]
            
            start = time.time()
            fcasts = []
            for t in range(cfg_test_dataset['seq_len']-backcast_len):
                if len(fcasts) > 0:
                    tgt_bcast = torch.cat([targets[:, t:backcast_len, :], torch.cat(fcasts, dim=1)], dim=1)
                else:
                    tgt_bcast = targets[:, t:backcast_len, :]
                
                inp_bcast = inputs[:, t:t+backcast_len, :]
                inp_fcast = inputs[:, t+backcast_len, :].unsqueeze(1)

                rf_inputs = torch.cat([inp_bcast.reshape(inputs.shape[0], -1), tgt_bcast.reshape(tgt_bcast.shape[0], -1), inp_fcast.reshape(inputs.shape[0], -1)], dim=-1)
                
                fcast = model.predict(rf_inputs.numpy())
                fcasts.append(torch.from_numpy(fcast[None, :, None]))
            fcasts = torch.cat(fcasts, dim=1)

            inf_times.append(time.time() - start)
            
            # If the targets were normalized during training, here at test time we have to unnormalize them
            bcast_t = dataset_test.unnormalize_data(tgt_bcast, dataset_test.normalization[2], norm_targets)
            fcasts = dataset_test.unnormalize_data(fcasts, dataset_test.normalization[2], norm_targets)
            fcast_t = dataset_test.unnormalize_data(tgt_fcast, dataset_test.normalization[2], norm_targets)
            inputs = dataset_test.unnormalize_data(inputs, dataset_test.normalization[0], norm_inputs)
            exog = dataset_test.unnormalize_data(exog, dataset_test.normalization[1], norm_exog)

            inps.append(inputs.detach().cpu())
            exogs.append(exog.detach().cpu())
            bcast_tgts.append(bcast_t.detach().cpu())
            forecasts.append(fcasts.detach().cpu())
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