import torch
import torch.nn as nn

from typing import Optional

import numpy as np

from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from utils.data.treenet_dataframe import TreeNetDataFrame
from ml.data.treenet_dataset import TreeNetTemporalDataset

from ml.models.tft.model import TemporalFusionTransformer

import captum
from captum.attr import GradientShap

import argparse
from pathlib import Path

import h5py

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Explain TFT Predictions',
    description='Explain predictions of TFT model using Gradient SHAP method.',
)

arg_parser.add_argument('--config', type=str, help='Path to the configuration file.')   
arg_parser.add_argument('--model_ckpt', type=str, help='Path to the model checkpoint.')   
arg_parser.add_argument('--output_path', type=str, help='Path to where the outputs should be stored.')
arg_parser.add_argument('--series_id', type=int, help='Which tree series to explain.')

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
        cache_path=None,
        input_feats=cfg_train_dataset['input_feats'],
        exog_feats=cfg_train_dataset['exog_feats'],
        target_feats=cfg_train_dataset['target_feats'],
        seq_len=cfg_train_dataset['seq_len'],
        max_dist_timesteps=cfg_train_dataset['max_dist_timesteps'],
        selected_input_feats=cfg_train_dataset['selected_input_feats'],
        normalization=cfg_train_dataset['normalization'],
        norm_inputs=cfg_train_dataset['norm_inputs'],
        norm_exog=cfg_train_dataset['norm_exog'],
        norm_targets=cfg_train_dataset['norm_targets'],
        target_only_growth=cfg_train_dataset['target_only_growth'],
        return_timestamp=True
    )

    if len(dataset_train) == 0:
        cfg_train_dataset = cfg['data']['init_args']['val_dataset']['init_args']
        dataset_train = TreeNetTemporalDataset(
            treenet_data_frame=TreeNetDataFrame(
                data_file_path=cfg_train_dataset['treenet_data_frame']['init_args']['data_file_path'],
                metadata_file_path=cfg_train_dataset['treenet_data_frame']['init_args']['metadata_file_path'],
                metadata_filter_fn=cfg_train_dataset['treenet_data_frame']['init_args']['metadata_filter_fn'],
                treedata_filter_fn=cfg_train_dataset['treenet_data_frame']['init_args']['treedata_filter_fn'],
                cache_path=cfg_train_dataset['treenet_data_frame']['init_args']['cache_path'],
                data_filter_fn='ts.dt.year == 2021 and series_id == %i' % args.series_id,
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
            cache_path=None,
            input_feats=cfg_train_dataset['input_feats'],
            exog_feats=cfg_train_dataset['exog_feats'],
            target_feats=cfg_train_dataset['target_feats'],
            seq_len=cfg_train_dataset['seq_len'],
            max_dist_timesteps=cfg_train_dataset['max_dist_timesteps'],
            selected_input_feats=cfg_train_dataset['selected_input_feats'],
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
        cache_path=None,
        input_feats=cfg_test_dataset['input_feats'],
        exog_feats=cfg_test_dataset['exog_feats'],
        target_feats=cfg_test_dataset['target_feats'],
        seq_len=cfg_test_dataset['seq_len'],
        max_dist_timesteps=cfg_test_dataset['max_dist_timesteps'],
        selected_input_feats=cfg_test_dataset['selected_input_feats'],
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

    with open('model.txt', 'w') as f:
        print(model, file=f) 

    data = torch.load(args.model_ckpt, map_location=torch.device('cpu'))                 # Load Pytorch Lightning checkpoint file
    state_dict = {k.partition('model.')[2]: v for k,v in data['state_dict'].items()}    # Extract state_dict and remove 'model' prefix from weights added by Pytorch Lightning
    model.load_state_dict(state_dict)                                                   # Load state_dict into the model
    model.eval()
    model = model.to(torch.float32)

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
    
    for m in model_wrapper.modules():
        if isinstance(m, nn.LSTM):
            m.train()

    norm_inputs_gpu = [dataset_test.norm_inputs[0].to('cuda'), dataset_test.norm_inputs[1].to('cuda')]
    norm_exog_gpu = [dataset_test.norm_exog[0].to('cuda'), dataset_test.norm_exog[1].to('cuda')]
    norm_targets_gpu = [dataset_test.norm_targets[0].to('cuda'), dataset_test.norm_targets[1].to('cuda')]

    backcast_len = cfg['model']['init_args']['backcast_len']
    forecast_len = cfg['model']['init_args']['forecast_len']

    base_inputs, base_exog, base_targets, _, _, _ = next(iter(dataloader_train))
    base_tgt_bcast = base_targets[:, :backcast_len, :]
    # defining model input tensors
    n_baselines = 64
    base_inputs_all = []
    for i in range(n_baselines):
        print('Preparing baseline %i / %i...' % (i+1, n_baselines))
        base_inputs, base_exog, base_targets, _, _, _ = next(iter(dataloader_train))

        # defining model input tensors
        base_inputs_cat = torch.cat([base_inputs, torch.cat([base_targets[:, :backcast_len, :], torch.zeros_like(base_targets[:, backcast_len:, :])], dim=1)], dim=-1)
        base_inputs_cat = torch.tensor(base_inputs_cat, requires_grad=True)
    
        base_inputs_all.append(torch.mean(base_inputs_cat, dim=0, keepdim=True))

    base_inputs_all = torch.cat(base_inputs_all, dim=0).to(torch.float32).to('cuda')

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

        tgt_bcast = targets[:, :backcast_len, :].to(torch.float32)
        tgt_fcast = targets[:, -forecast_len:, :].to(torch.float32)

        inputs_all = torch.cat([inputs, torch.cat([targets[:, :backcast_len, :], torch.zeros_like(targets[:, backcast_len:, :])], dim=1)], dim=-1)
        inputs_all = torch.tensor(inputs_all).to(torch.float32).to('cuda')

        all_attributions = []
        all_vars = []
        tstamps = []

        inputs_all_rep = inputs_all.repeat_interleave(forecast_len, dim=0)

        chunk = 12  # tune to your GPU
        attr_chunks = []
        for h0 in range(0, forecast_len, chunk):
            h = torch.arange(h0, min(h0 + chunk, forecast_len), device=device)
            attr_chunks.append(dlshap.attribute(
                inputs=inputs_all.expand(len(h), -1, -1).contiguous(),
                baselines=base_inputs_all,
                stdevs=0.025,
                n_samples=32,
                target=h,
            ))
        attributions = torch.cat(attr_chunks, dim=0)   # [forecast_len, T, F]

        attrs_inps_all = attributions[..., :-1]
        inps_all = inputs_all[..., :-1] # They're all the same, just pick one in the first dimension

        attrs_tgt_all = attributions[..., -1:]
        tgts_all = torch.cat([tgt_bcast, tgt_fcast], dim=1).to('cuda')
        
        attrs_all = torch.cat([attrs_inps_all, attrs_tgt_all], dim=2)
        inps_all = dataset_test.unnormalize_data(inps_all, dataset_test.normalization[0], norm_inputs_gpu)
        tgts_all = dataset_test.unnormalize_data(tgts_all, dataset_test.normalization[2], norm_targets_gpu)
        vars_all = torch.cat([inps_all, tgts_all], dim=2)

        with torch.no_grad():
            pred = model_wrapper(inputs_all)
            pred = dataset_test.unnormalize_data(pred, dataset_test.normalization[2], norm_targets_gpu)

        series_attributions.append(attrs_all.detach().cpu().numpy())
        series_vars.append(vars_all.detach().cpu().numpy())
        series_preds.append(pred.detach().cpu().numpy())
        series_tstamps.append(tstamp.detach().cpu().numpy())

    series_attributions = np.stack(series_attributions)
    series_vars = np.stack(series_vars)
    series_preds = np.stack(series_preds)
    series_tstamps = np.stack(series_tstamps)
    
    with h5py.File(explain_path.joinpath('%s_%i.h5' % (Path(args.config).stem, args.series_id)), "w") as f:
        f.create_dataset('all_attributions', data=series_attributions.astype(np.float32), dtype='f4')
        f.create_dataset('all_vars', data=series_vars.astype(np.float32), dtype='f4')
        f.create_dataset('backcast_len', data=backcast_len, dtype="i4")
        f.create_dataset('forecast_len', data=forecast_len, dtype="i4")
        f.create_dataset('input_feats', data=dataset_test.input_feats, dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset('target_feats', data=dataset_test.target_feats, dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset('pred', data=series_preds.astype(np.float32), dtype='f4')
        f.create_dataset('tstamp', data=series_tstamps.astype(np.float32), dtype='f4')

print('Done')