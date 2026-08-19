import torch
import numpy as np
import pandas as pd

from scripts.viz.utils.data_utils import load_attributions

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Explain Variable Contribution',
    description='Explain contribution of different variables to the change in TWD.',
)

arg_parser.add_argument('--data_path', type=str, help='Path to the dataset folder.')   
arg_parser.add_argument('--output_path', type=str, help='Path to where the outputs should be stored.')
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
    # Quercus Petraea: ['120', '853']
    # Pinus sylvestris: ['1', '3', '166', '696', '1208', '1210', '1270', '1276', '1365', '1396']
    # Picea abies:  ['22', '27', '557', '686', '1180', '1206', '1268', '1381']
    # Fagus sylvatica: ['18', '156', '160', '687', '849', '855', '1094', '1169', '1218', '1385']
    # Abies alba: ['28', '693', '1163', '1382'] 

    # Load model evaluation data
    series_ids_all =  [
        ['1351'],#['172', '1099_0', '1099_1', '1351', '1369'],
        ['853'],#['120', '853'],
        ['166'],#['1', '3', '166', '696', '1208', '1210', '1270', '1276', '1365', '1396'],
        ['557'],#['21', '27', '557', '686', '1180', '1206', '1268', '1381'],
        ['160'],#['18', '156', '160', '687', '849', '855', '1094', '1169', '1218', '1385'],
        ['1382'],#['28', '693', '1163', '1382'],
    ]

    series_names = ['Quercus pubescens', 'Quercus petraea', 'Pinus sylvestris', 'Picea abies', 'Fagus sylvatica', 'Abies alba']

    input_feat_names = {
        'rad': 'Solar Radiation [W/m²]',
        'vpd': 'Vapor Pressure Deficit [kPa]',
        'swp_abs_log1p': 'Soil Water Potential [pF]',
        'temp': 'Air Temperature [°C]',
        'rh': 'Relative Humidity [%]',
        'total_precip_log1p': 'Total Precipitation [mm]'
    }

    series_colors = {
        'Quercus pubescens': 'rgba(93, 164, 214, 0.75)', 
        'Quercus petraea': 'rgba(148, 103, 189, 0.75)',
        'Pinus sylvestris': 'rgba(255, 144, 14, 0.75)', 
        'Picea abies': 'rgba(44, 160, 101, 0.75)', 
        'Fagus sylvatica':  'rgba(255, 65, 54, 0.75)', 
        'Abies alba': 'rgba(127, 96, 0, 0.75)',
    }

    attrs_series = []
    vars_series = []
    input_features = None
    for sidx, series_ids in enumerate(series_ids_all):
        print('Processing series ids %i...' % sidx)
        attr_flat = [torch.tensor([]) for _ in range(len(input_feat_names))]
        vars_flat = [torch.tensor([]) for _ in range(len(input_feat_names))]
        for serid in series_ids:
            dser = load_attributions(Path(args.output_path).joinpath('explain'), args.model, [serid], only_attribution=False)
            tstamp = dser['tstamps']
            forecast_len = dser['forecast_len']
            backcast_len = dser['backcast_len']
            pred = dser['preds']
            all_attributions = dser['attributions']
            all_vars = dser['vars'].unsqueeze_(1).repeat(1, all_attributions.shape[1], 1, 1)
            input_features = dser['input_feats']

            all_vars[..., 0] = torch.expm1(all_vars[..., 0]) # total precipitation (convert from log1p to mm)
            all_vars[..., 3] = torch.log10(torch.expm1(all_vars[..., 3]) * 10) # swp (convert from log1p to pF)
            
            for feat_idx in range(len(input_feat_names)):
                flat_vars = all_vars[:, :, -forecast_len:, feat_idx].mean(dim=1).flatten()
                unique_vars, unique_idxs = flat_vars.flatten().unique(return_inverse=True)

                flat_feat_futr_attrs = all_attributions[:, :, -forecast_len:, feat_idx].mean(dim=2).flatten()
                avg_feat_futr_attrs = torch.zeros((unique_vars.shape[0], ), dtype=all_vars.dtype)
                avg_feat_futr_attrs.index_reduce_(dim=0, index=unique_idxs, source=flat_feat_futr_attrs, reduce='mean')

                vars_flat[feat_idx] = torch.cat([vars_flat[feat_idx], unique_vars], dim=0)
                attr_flat[feat_idx] = torch.cat([attr_flat[feat_idx], avg_feat_futr_attrs], dim=0)
            
        attrs_ser = []
        vars_ser = []
        for idx in range(len(attr_flat)):
            x = vars_flat[idx]
            y = attr_flat[idx]
            x_grid = np.linspace(x.min(), x.max(), 5)
            kernel = RBF(length_scale=2.0) + WhiteKernel(noise_level=1.0)
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            gpr.fit(x[:, None], y)
            y_fit, y_std = gpr.predict(x_grid[:, None], return_std=True)
            attrs_ser.append(y_fit)
            vars_ser.append(x_grid)

        attrs_series.append(attrs_ser)
        vars_series.append(vars_ser)
        del dser
        del attr_flat
        del vars_flat
    print('Done.')

    input_features = [input_feat_names[feat] for feat in input_features]

    fig = make_subplots(rows=2, cols=3, shared_xaxes=False, shared_yaxes=True, subplot_titles=input_features, y_title='Average Contribution', horizontal_spacing = 0.05, vertical_spacing = 0.1)

    y_axis_default = dict(
            range=[-0.005,0.005],
            tickvals=[-0.0025, 0.0025],
            ticktext=['negative', 'positive'],
        )

    for feature_index in range(6):
        attrs_all = []
        vars_all = []
        for attr_ser in attrs_series:
            attrs_all.append(attr_ser[feature_index])
        for vars_ser in vars_series:
            vars_all.append(vars_ser[feature_index])
        
        # Use x instead of y argument for horizontal plot
        for i in range(len(attrs_all)):
            fig.add_trace(go.Scatter(
                x=vars_all[i],
                y=attrs_all[i],
                name=series_names[i],
                marker_color=series_colors[series_names[i]],
                line_width=5,
                showlegend=feature_index==0,
            ), row=(feature_index//3)+1, col=(feature_index%3)+1)
            fig.update_annotations(font_size=22)
            fig.update_yaxes(y_axis_default, row=(feature_index//3)+1, col=(feature_index%3)+1)

    fig.update_layout(
        width=1600, 
        height=1200, 
        margin=dict(
            l=200,
            r=50,
            t=50,
            b=50
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font_size=22
        ),
        font=dict(size=22),
    )

    fig['layout']['annotations'][-1].xshift = -150

    fig.write_image(figures_path.joinpath('explain_vars_%s.png' % args.model))