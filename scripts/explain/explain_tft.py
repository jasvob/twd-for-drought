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

from ml.models.tft.model import TemporalFusionTransformer

import captum
from captum.attr import GradientShap

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Explain TFT Predictions',
    description='Explain predictions of TFT model using Gradient SHAP method.',
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

    class ModelWrapper(nn.Module):
        def __init__(self, model : TemporalFusionTransformer) -> None:
            super(ModelWrapper, self).__init__()

            self.model = model

        def forward(self, inputs_all : torch.Tensor, hor_idx : Optional[int] = None) -> torch.Tensor:
            preds = self.model(inputs_all, None, None, None)

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

    # defining model input tensors
    n_baselines = 64
    base_inputs_all = []
    for i in range(n_baselines):
        base_inputs, base_exog, base_targets, _, _, _ = next(iter(dataloader_train))

        base_inputs_cat = torch.cat([base_inputs, torch.cat([base_targets[:, :backcast_len, :], torch.zeros_like(base_targets[:, backcast_len:, :])], dim=1)], dim=-1)
        base_inputs_cat = torch.tensor(base_inputs_cat, requires_grad=True)
    
        base_inputs_all.append(torch.mean(base_inputs_cat, dim=0, keepdim=True))

    base_inputs_all = torch.cat(base_inputs_all, dim=0).to('cuda')

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

        targets = targets.to('cuda')
        inputs = inputs.to('cuda')
        exog = exog.to('cuda')

        tgt_bcast = targets[:, :backcast_len, :]

        inputs_all = torch.cat([inputs, torch.cat([targets[:, :backcast_len, :], torch.zeros_like(targets[:, backcast_len:, :])], dim=1)], dim=-1)
        inputs_all = torch.tensor(inputs_all, requires_grad=True).to('cuda')

        all_attributions = []
        all_vars = []
        tstamps = []
        btsz = 8
        inputs_all = inputs_all.repeat(btsz, 1, 1)
        for tgt_idx in range(forecast_len // btsz):
            #print(tgt_idx, tgt_idx*btsz, (tgt_idx+1)*btsz)
            with torch.backends.cudnn.flags(enabled=False):
                attributions, approximation_error = dlshap.attribute(inputs_all,
                                                                baselines=base_inputs_all,
                                                                stdevs=0.025, 
                                                                n_samples=32,
                                                                target=torch.tensor([i for i in range(tgt_idx*btsz, (tgt_idx+1)*btsz)]).to('cuda'),
                                                                return_convergence_delta=True)
                                                            
            #print(approximation_error)
            attrs_inps_all = attributions[..., :-1]
            inps_all = inputs_all[..., :-1] # They're all the same, just pick one in the first dimension

            attrs_tgt_all = attributions[..., -1:]
            tgts_all = inputs_all[..., -1:] # They're all the same, just pick one in the first dimension

            attrs_all = torch.cat([attrs_inps_all, attrs_tgt_all], dim=2)
            inps_all = dataset_test.unnormalize_data(inps_all, dataset_test.normalization[0], norm_inputs_gpu)
            tgts_all = dataset_test.unnormalize_data(tgts_all, dataset_test.normalization[2], norm_targets_gpu)
            vars_all = torch.cat([inps_all, tgts_all], dim=2)

            all_attributions.append(attrs_all.detach().cpu())
            all_vars.append(vars_all.detach().cpu())

        all_attributions = torch.cat(all_attributions, dim=0).unsqueeze(1)
        all_vars = torch.cat(all_vars, dim=0).unsqueeze(1)

        pred = model_wrapper(inputs_all[:1, :, :])
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