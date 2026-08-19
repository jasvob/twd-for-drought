import torch

import sys
sys.path.append('../')

import numpy as np

from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from utils.data.treenet_dataframe import TreeNetDataFrame
from ml.data.treenet_dataset import TreeNetTemporalDataset

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Compute normalization constants',
    description='Compute normalization constants for a dataset.',
)

arg_parser.add_argument('--config', type=str, help='Path to the configuration file.')   
arg_parser.add_argument('--split', type=str, help='One of [train, val, test].')

if __name__ == "__main__":
    args = arg_parser.parse_args()

    cfg = OmegaConf.load(args.config)

    cfg_model = cfg['model']['init_args']['model']['init_args']
    cfg_dataset = cfg['data']['init_args']['%s_dataset' % args.split]['init_args']

    dataset = TreeNetTemporalDataset(
        treenet_data_frame=TreeNetDataFrame(
            data_file_path=cfg_dataset['treenet_data_frame']['init_args']['data_file_path'],
            metadata_file_path=cfg_dataset['treenet_data_frame']['init_args']['metadata_file_path'],
            metadata_filter_fn=cfg_dataset['treenet_data_frame']['init_args']['metadata_filter_fn'],
            treedata_filter_fn=cfg_dataset['treenet_data_frame']['init_args']['treedata_filter_fn'],
            cache_path=cfg_dataset['treenet_data_frame']['init_args']['cache_path'],
            data_filter_fn=cfg_dataset['treenet_data_frame']['init_args']['data_filter_fn'],
            min_trees_per_site=cfg_dataset['treenet_data_frame']['init_args']['min_trees_per_site'],
            equal_sequence_per_site=cfg_dataset['treenet_data_frame']['init_args']['equal_sequence_per_site'],
            interpolate_data=cfg_dataset['treenet_data_frame']['init_args']['interpolate_data'],
            drop_na_for_variables=cfg_dataset['treenet_data_frame']['init_args']['drop_na_for_variables'],
            drop_na_for_metadata=cfg_dataset['treenet_data_frame']['init_args']['drop_na_for_metadata'],
            resample_data=cfg_dataset['treenet_data_frame']['init_args']['resample_data'],
            filter_growth_period=cfg_dataset['treenet_data_frame']['init_args']['filter_growth_period'],
            filter_by_temperature=cfg_dataset['treenet_data_frame']['init_args']['filter_by_temperature'],
            filter_by_frost=cfg_dataset['treenet_data_frame']['init_args']['filter_by_frost'],
            filter_min_growth=cfg_dataset['treenet_data_frame']['init_args']['filter_min_growth'],
            filter_min_twd=cfg_dataset['treenet_data_frame']['init_args']['filter_min_twd'],
            merge_by_site=cfg_dataset['treenet_data_frame']['init_args']['merge_by_site'],
        ),
        input_feats=cfg_dataset['input_feats'],
        exog_feats=cfg_dataset['exog_feats'],
        target_feats=cfg_dataset['target_feats'],
        seq_len=cfg_dataset['seq_len'],
        max_dist_timesteps=cfg_dataset['max_dist_timesteps'],
        normalization=('none', 'none', 'none'),
        norm_inputs=cfg_dataset['norm_inputs'],
        norm_exog=cfg_dataset['norm_exog'],
        norm_targets=cfg_dataset['norm_targets'],
        target_only_growth=cfg_dataset['target_only_growth'],
        return_timestamp=False
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    inputs_tnet_all = []
    targets_tnet_all = []
    for i, data in enumerate(dataloader):
        if i % 100 == 0:
            print(i, ' of ', len(dataloader))
            sys.stdout.flush()
        inputs_tnet, exog, targets= data
        inputs_tnet_all.append(inputs_tnet)
        targets_tnet_all.append(targets)
    inputs_tnet_all = torch.cat(inputs_tnet_all, dim=0)
    targets_tnet_all = torch.cat(targets_tnet_all, dim=0)

    print(inputs_tnet_all.mean(dim=[0,1]).tolist())
    print(inputs_tnet_all.std(dim=[0,1]).tolist())

    print(targets_tnet_all.mean(dim=[0,1]).tolist())
    print(targets_tnet_all.std(dim=[0,1]).tolist())