import torch
import pandas as pd
import numpy as np

from scripts.viz.utils.data_utils import load_attributions
from scripts.viz.utils.plot_utils import plot_attributions_series

import plotly.graph_objects as go

import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Explain Covariates Contribution',
    description='Explain contribution of different future covariates to the change in TWD.',
)

arg_parser.add_argument('--data_path', type=str, help='Path to the dataset folder.')   
arg_parser.add_argument('--output_path', type=str, help='Path to where the outputs should be stored.')
arg_parser.add_argument('--series_ids', type=str, help='List of Series IDs to plot explanations for.')
arg_parser.add_argument('--model', type=str, default='nhits', help='One of [nhits, tft].')

if __name__ == "__main__":
    args = arg_parser.parse_args()

    # Create output folders if they don't exist
    figures_path = Path(args.output_path).joinpath('figures')
    figures_path.mkdir(parents=True, exist_ok=True)

    # Get model config file
    cfg = OmegaConf.load('configs/%s.yaml' % args.model)
    selected_input_feates = cfg['data']['init_args']['train_dataset']['init_args']['selected_input_feats']
    input_features = np.array(cfg['data']['init_args']['train_dataset']['init_args']['input_feats'])[selected_input_feates].tolist()
    input_feat_names = {
        'rad': 'rad',
        'vpd': 'vpd',
        'swp_abs_log1p': 'swp',
        'temp': 'temp',
        'rh': 'rh',
        'total_precip_log1p': 'prec'
    }
    input_features = [input_feat_names[feat] for feat in input_features]

    # Load tree metadata
    metadata_frame = pd.read_parquet(Path(args.data_path).joinpath('filt_metadata.parquet'))
    metadata_frame = metadata_frame.set_index('series_id')

    # Quercus pubescens: 172, 1099, 1351, 1369
    # Quercus petraea: 120, 853 
    # Pinus sylvestris: 1, 3, 166, 696, 1208, 1210, 1270, 1276, 1365, 1396
    # Picea abies: 21, 27, 221, 557, 686, 1180, 1206, 1268, 1381
    # Fagus sylvatica: 18, 156, 160, 687, 849, 855, 1094, 1112, 1162, 1169, 1218, 1385
    # Abies alba: 28, 693, 1163, 1382
    # Fraxinus excelsior: 1167, 1366

    # Load model evaluation data
    series_ids = [int(x) for x in args.ids.split(",")]

    unique_tstamps_all = []
    avg_preds_all = []
    avg_feat_futr_attrs_all = []
    subplot_titles = []

    for sid in series_ids:
        species_name = '_'.join(metadata_frame.loc[int(series_ids[0])][['tree_genus', 'tree_species']].to_list())
        data_ser = load_attributions(Path(args.output_path).joinpath('explain'), args.model, [sid], only_attribution=False)

        tstamp = data_ser['tstamps']
        pred = data_ser['preds']
        all_attributions = data_ser['attributions']
        forecast_len = data_ser['forecast_len']
        backcast_len = data_ser['backcast_len']

        minutes_per_2h = 86400 // 12
        tstamps_fcast = tstamp.unsqueeze(1) + torch.linspace(0, minutes_per_2h * (forecast_len - 1), forecast_len, dtype=torch.int64).unsqueeze(0)

        unique_tstamps, unique_idxs = tstamps_fcast.flatten().unique(return_inverse=True)
        flat_preds = pred.squeeze().flatten()
        avg_preds = torch.zeros_like(unique_tstamps, dtype=pred.dtype)
        avg_preds.index_reduce_(dim=0, index=unique_idxs, source=flat_preds, reduce='mean', include_self=False)

        flat_feat_attrs = all_attributions[:, :, :, :].mean(dim=2).flatten(end_dim=-2)
        avg_feat_attrs = torch.zeros((unique_tstamps.shape[0], flat_feat_attrs.shape[1]), dtype=pred.dtype)
        avg_feat_attrs.index_reduce_(dim=0, index=unique_idxs, source=flat_feat_attrs, reduce='mean', include_self=False)

        flat_feat_futr_attrs = all_attributions[:, :, -forecast_len:, :].mean(dim=2).flatten(end_dim=-2)
        avg_feat_futr_attrs = torch.zeros((unique_tstamps.shape[0], flat_feat_futr_attrs.shape[1]), dtype=pred.dtype)
        avg_feat_futr_attrs.index_reduce_(dim=0, index=unique_idxs, source=flat_feat_futr_attrs, reduce='mean', include_self=False)

        flat_feat_past_attrs = all_attributions[:, :, :backcast_len, :].mean(dim=2).flatten(end_dim=-2)
        avg_feat_past_attrs = torch.zeros((unique_tstamps.shape[0], flat_feat_past_attrs.shape[1]), dtype=pred.dtype)
        avg_feat_past_attrs.index_reduce_(dim=0, index=unique_idxs, source=flat_feat_past_attrs, reduce='mean', include_self=False)

        unique_tstamps_all.append(unique_tstamps)
        avg_preds_all.append(avg_preds)
        avg_feat_futr_attrs_all.append(avg_feat_futr_attrs)
        subplot_titles.append('%s (%s)' % (species_name, sid))

    fig = plot_attributions_series(unique_tstamps_all, avg_preds_all, avg_feat_futr_attrs_all, input_features, subplot_titles, row_height=400)
    fig.write_image(figures_path.joinpath('explain_series_%s.png' % args.model))