import torch
import numpy as np
import pandas as pd

from scripts.viz.utils.data_utils import load_attributions

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.api as sm # to build a LOWESS model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.stats import binned_statistic

import argparse
from pathlib import Path
from omegaconf import OmegaConf

from concurrent.futures import ProcessPoolExecutor

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

    # Quercus pubescens: ['172', '1099', '1351', '1369']
    # Quercus Petraea: ['120', '853']
    # Pinus sylvestris: ['1', '3', '166', '696', '1208', '1210', '1270', '1276', '1365', '1396']
    # Picea abies:  ['21', '27', '221', '557', '686', '1180', '1206', '1268', '1381']
    # Fagus sylvatica: '18', '156', '160', '687', '849', '855', '1094', '1112', '1162', '1169', '1218', '1385']
    # Abies alba: ['28', '693', '1163', '1382'] 
    # Fraxinus excelsior ['1167', '1366'] 

    # Load model evaluation data
    series_ids_all =  [
        ['1167'],#['1167', '1366'],
        ['1382'],#['28', '693', '1163', '1382'],
        ['1351'],#['172', '1099', '1351', '1369'],
        ['853'],#['120', '853'],
        ['166'],#['1', '3', '166', '696', '1208', '1210', '1270', '1276', '1365', '1396'],
        ['557'],#['21', '27', '221', '557', '686', '1180', '1206', '1268', '1381'],
        ['160']#['18', '156', '160', '687', '849', '855', '1094', '1112', '1162', '1169', '1218', '1385'],
    ]

    series_names = ['Fraxinus excelsior', 'Abies alba', 'Quercus pubescens', 'Quercus petraea', 'Pinus sylvestris', 'Picea abies', 'Fagus sylvatica']

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
        'Fraxinus excelsior': 'rgba(23, 190, 207, 0.75)',
    }
    series_fill_colors = {
        'Quercus pubescens': 'rgba(93, 164, 214, 0.2)', 
        'Quercus petraea': 'rgba(148, 103, 189, 0.2)',
        'Pinus sylvestris': 'rgba(255, 144, 14, 0.2)', 
        'Picea abies': 'rgba(44, 160, 101, 0.2)', 
        'Fagus sylvatica':  'rgba(255, 65, 54, 0.2)', 
        'Abies alba': 'rgba(127, 96, 0, 0.2)',
        'Fraxinus excelsior': 'rgba(23, 190, 207, 0.2)',
    }

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

    #####################
    def process_attributions(serid):
        print('Processing series %s...' % serid)

        dser = load_attributions(Path(args.output_path).joinpath('explain'), args.model, [serid], only_attribution=False)
        #tstamp = dser['tstamps']
        forecast_len = dser['forecast_len']
        #backcast_len = dser['backcast_len']
        #pred = dser['preds']
        all_attributions = dser['attributions']
        all_vars = dser['vars']

        precip_idx = input_features.index('total_precip_log1p')
        swp_idx = input_features.index('swp_abs_log1p')
        all_vars[..., precip_idx] = torch.expm1(all_vars[..., precip_idx]) # total precipitation (convert from log1p to mm)
        all_vars[..., swp_idx] = torch.log10(torch.expm1(torch.clamp(all_vars[..., swp_idx], min=1e-3)) * 10) # swp (convert from log1p to pF)

        vars_feat_flat = []
        attr_feat_flat = []
        for feat_idx in range(len(input_features)):
            # Take the first index (0) at the dimension 1, as at every index you will find the same input
            flat_vars = all_vars[:, 0, -forecast_len:, feat_idx].flatten()
            unique_vars, unique_idxs = flat_vars.flatten().unique(return_inverse=True)

            flat_feat_futr_attrs = all_attributions[:, :, -forecast_len:, feat_idx].mean(dim=2).flatten()
            avg_feat_futr_attrs = torch.zeros((unique_vars.shape[0], ), dtype=all_vars.dtype)
            avg_feat_futr_attrs.index_reduce_(dim=0, index=unique_idxs, source=flat_feat_futr_attrs, reduce='mean', include_self=False)

            vars_feat_flat.append(unique_vars)
            attr_feat_flat.append(avg_feat_futr_attrs) 

        print('%s Done.' % serid)

        return vars_feat_flat, attr_feat_flat, input_features
    #####################

    attrs_series = []
    attrs_series_std = []
    vars_series = []
    pts_x_series = []
    pts_y_series = []
    for sidx, series_ids in enumerate(series_ids_all):
        print('Processing species %s...' % series_names[sidx])
        attr_flat_all = [[] for _ in range(len(input_features))]
        vars_flat_all = [[] for _ in range(len(input_features))]
        with ProcessPoolExecutor(max_workers=8) as ex:
            for vars_list, attr_list, input_feats in ex.map(process_attributions, series_ids):
                for feat_idx in range(len(input_features)):
                    vars_flat_all[feat_idx].append(vars_list[feat_idx])
                    attr_flat_all[feat_idx].append(attr_list[feat_idx]) 
                    input_features = input_feats
                    
        vars_flat = [torch.cat(vf) for vf in vars_flat_all]
        attr_flat = [torch.cat(af) for af in attr_flat_all]

        print('Fitting GPs to features ...')
        attrs_ser = []
        attrs_ser_std = []
        vars_ser = []
        pts_x_ser, pts_y_ser = [], []    
        for idx in range(len(attr_flat)):
            print('Feature %i ...' % idx)
            x = vars_flat[idx]
            y = attr_flat[idx]

            rng = np.random.default_rng(0)
            keep = rng.choice(len(x), size=min(len(x), 3000), replace=False)
            pts_x_ser.append(x[keep])
            pts_y_ser.append(y[keep])


            xm, xs = x.mean(), x.std()
            x = (x - xm) / xs

            n_bins = 1000
            means, edges, _ = binned_statistic(x.numpy(), y.numpy(), statistic="mean", bins=n_bins)
            counts, _, _    = binned_statistic(x.numpy(), y.numpy(), statistic="count", bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            filter_pts = counts >= np.mean(counts)

            x_grid = np.linspace(centers[filter_pts].min(), centers[filter_pts].max(), 6)#10)

            kernel = ConstantKernel(1.0, (1e-3, 1e2)) * RBF(length_scale=2.0, length_scale_bounds=(0.05, 20.0)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-4, 10.0))
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=0, random_state=0)
            gpr.fit(centers[filter_pts, None].astype(np.float64), means[filter_pts].astype(np.float64))
            y_fit, y_std = gpr.predict(x_grid[:, None].astype(np.float64), return_std=True)
            noise = gpr.kernel_.k2.noise_level * gpr._y_train_std**2   # undo normalize_y
            y_std = np.sqrt(np.maximum(y_std**2 - noise, 0)) 

            x_grid = (x_grid * xs.numpy()) + xm.numpy()

            attrs_ser.append(y_fit)
            attrs_ser_std.append(y_std)
            vars_ser.append(x_grid)

        attrs_series.append(attrs_ser)
        attrs_series_std.append(attrs_ser_std)
        vars_series.append(vars_ser)
        pts_x_series.append(pts_x_ser)       
        pts_y_series.append(pts_y_ser)
        del attr_flat
        del vars_flat
        print('Done.')

    input_features = [input_feat_names[feat] for feat in input_features]

    fig = make_subplots(rows=1, cols=3, shared_xaxes=False, shared_yaxes=True, subplot_titles=input_features, y_title='Average Contribution', horizontal_spacing = 0.05, vertical_spacing = 0.1)

    y_axis_default = dict(
            range=[-0.005,0.005],
            tickvals=[-0.0025, 0.0025],
            ticktext=['negative', 'positive'],
        )

    for feature_index in range(len(input_features)):
        attrs_all = []
        attrs_std_all = []
        vars_all = []
        for attr_ser in attrs_series:
            attrs_all.append(attr_ser[feature_index])
        for attr_ser_std in attrs_series_std:
            attrs_std_all.append(attr_ser_std[feature_index])
        for vars_ser in vars_series:
            vars_all.append(vars_ser[feature_index])
        
        # Use x instead of y argument for horizontal plot
        for i in range(len(attrs_all)):
            fig.add_trace(go.Scatter(
                x=pts_x_series[i][feature_index],
                y=pts_y_series[i][feature_index],
                mode='markers',
                marker=dict(size=2, color='gray', opacity=0.25,
                            line_width=0),
                showlegend=False,
                hoverinfo='skip',
            ), row=(feature_index//3)+1, col=(feature_index%3)+1, secondary_y=False)
        for i in range(len(attrs_all)):
            fig.add_trace(go.Scatter(
                x=vars_all[i],
                y=attrs_all[i],
                name=series_names[i],
                marker_color=series_colors[series_names[i]],
                line_width=5,
                showlegend=feature_index==0,
            ), row=(feature_index//3)+1, col=(feature_index%3)+1)
            fig.add_trace(go.Scatter(
                x=vars_all[i],
                y=attrs_all[i] - attrs_std_all[i],
                mode='lines',
                marker_color=series_colors[series_names[i]],
                line_width=1,
                name=series_names[i],
                showlegend=False
            ), row=(feature_index//3)+1, col=(feature_index%3)+1)
            fig.add_trace(go.Scatter(
                x=vars_all[i],
                y=attrs_all[i] + attrs_std_all[i],
                mode='lines',
                marker_color=series_colors[series_names[i]],
                line_width=1,
                name=series_names[i],
                fillcolor=series_fill_colors[series_names[i]],
                fill='tonexty',
                showlegend=False
            ), row=(feature_index//3)+1, col=(feature_index%3)+1)
            fig.update_annotations(font_size=22)
            fig.update_yaxes(y_axis_default, row=(feature_index//3)+1, col=(feature_index%3)+1)

    fig.update_layout(
        width=1600, 
        height=600, 
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