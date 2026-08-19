import torch
import pandas as pd

from scripts.viz.utils.data_utils import load_attributions
from scripts.viz.utils.plot_utils import plot_attributions_series

import plotly.graph_objects as go

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Explain Covariates Contribution',
    description='Explain contribution of different future covariates to the change in TWD.',
)

arg_parser.add_argument('--data_path', type=str, help='Path to the dataset folder.')   
arg_parser.add_argument('--output_path', type=str, help='Path to where the outputs should be stored.')
arg_parser.add_argument('--series_id', type=int, help='Series ID to plot explanations for.')
arg_parser.add_argument('--model', type=str, default='nhits', help='One of [nhits, tft].')

if __name__ == "__main__":
    args = arg_parser.parse_args()

    # Create output folders if they don't exist
    figures_path = Path(args.output_path).joinpath('figures')
    figures_path.mkdir(parents=True, exist_ok=True)

    # Load tree metadata
    metadata_frame = pd.read_parquet(Path(args.data_path).joinpath('filt_metadata.parquet'))
    metadata_frame = metadata_frame.set_index('series_id')

    # Quercus pubescens: ['172', '1099_0', '1099_1', '1351', '1369']
    # Pinus sylvestris: ['1', '3', '166', '696', '1208', '1210', '1270', '1276', '1365', '1396']
    # Picea abies:  ['22', '27', '557', '686', '1180', '1206', '1268', '1381']
    # Fagus sylvatica: ['18', '156', '160', '687', '849', '855', '1094', '1169', '1218', '1385']
    # Abies alba: ['28', '693', '1163', '1382'] 

    # Load model evaluation data
    series_ids =  [args.series_id] 
    species_name = '_'.join(metadata_frame.loc[int(series_ids[0])][['tree_genus', 'tree_species']].to_list())
    data_ser = load_attributions(Path(args.output_path).joinpath('explain'), args.model, series_ids, only_attribution=False)

    tstamp = data_ser['tstamps']
    pred = data_ser['preds']
    all_attributions = data_ser['attributions']
    all_vars = data_ser['vars'].unsqueeze_(1).repeat(1, all_attributions.shape[1], 1, 1)
    forecast_len = data_ser['forecast_len']
    backcast_len = data_ser['backcast_len']

    minutes_per_2h = 86400 // 12
    tstamps_fcast = tstamp.unsqueeze(1) + torch.linspace(0, minutes_per_2h * (forecast_len - 1), forecast_len, dtype=torch.int64).unsqueeze(0)

    unique_tstamps, unique_idxs = tstamps_fcast.flatten().unique(return_inverse=True)
    flat_preds = pred.squeeze().flatten()
    avg_preds = torch.zeros_like(unique_tstamps, dtype=pred.dtype)
    avg_preds.index_reduce_(dim=0, index=unique_idxs, source=flat_preds, reduce='mean')

    flat_feat_attrs = all_attributions[:, :, :, :].mean(dim=2).flatten(end_dim=-2)
    avg_feat_attrs = torch.zeros((unique_tstamps.shape[0], flat_feat_attrs.shape[1]), dtype=pred.dtype)
    avg_feat_attrs.index_reduce_(dim=0, index=unique_idxs, source=flat_feat_attrs, reduce='mean')

    flat_feat_futr_attrs = all_attributions[:, :, -forecast_len:, :].mean(dim=2).flatten(end_dim=-2)
    avg_feat_futr_attrs = torch.zeros((unique_tstamps.shape[0], flat_feat_futr_attrs.shape[1]), dtype=pred.dtype)
    avg_feat_futr_attrs.index_reduce_(dim=0, index=unique_idxs, source=flat_feat_futr_attrs, reduce='mean')

    flat_feat_past_attrs = all_attributions[:, :, :backcast_len, :].mean(dim=2).flatten(end_dim=-2)
    avg_feat_past_attrs = torch.zeros((unique_tstamps.shape[0], flat_feat_past_attrs.shape[1]), dtype=pred.dtype)
    avg_feat_past_attrs.index_reduce_(dim=0, index=unique_idxs, source=flat_feat_past_attrs, reduce='mean')

    flat_vars = all_vars[:, :, :, :].mean(dim=2).flatten(end_dim=-2)
    avg_flat_vars = torch.zeros((unique_tstamps.shape[0], flat_vars.shape[1]), dtype=pred.dtype)
    avg_flat_vars.index_reduce_(dim=0, index=unique_idxs, source=flat_vars, reduce='mean')

    input_feat_names = {
        'rad': 'rad',
        'vpd': 'vpd',
        'swp_abs_log1p': 'swp',
        'temp': 'temp',
        'rh': 'rh',
        'total_precip_log1p': 'prec'
    }
    input_features = [input_feat_names[feat] for feat in data_ser['input_feats']]

    fig = plot_attributions_series(unique_tstamps, avg_preds, avg_feat_futr_attrs, input_features)
    fig.write_image(figures_path.joinpath('explain_%s_%s_%s.png' % (args.model, species_name.lower(), series_ids[0])))