import torch

import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from scripts.viz.utils.data_utils import load_data
from scripts.viz.utils.plot_utils import plot_forecast_sample_overlay

import plotly.graph_objects as go

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Plot Forecasts',
    description='Plot qualitative examples of forecasts for a model.',
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
        
    # Load model evaluation data
    data_ser = load_data(Path(args.output_path).joinpath('eval'), args.model)

    series_samples = {
        160: [200, 300],    # Fagus sylvatica
        166: [670, 850],    # Pinus sylvestris
        557: [200, 380],    # Picea abies
        1099: [200, 600],   # Quercus pubescens
        1382: [50, 350],    # Abies alba
        853: [75, 550],     # Quercus petraea
    }

    for i, (series_idx, sample_list) in enumerate(series_samples.items()):
        mask_ser = (data_ser['series_ids'] == series_idx)
        species_name = '_'.join(metadata_frame.loc[series_idx][['tree_genus', 'tree_species']].to_list())

        for j, sample_idx in enumerate(sample_list):
            fig = go.Figure()

            plot_forecast_sample_overlay(data_ser, mask_ser, sample_idx, fig, (41, 101, 169), show_target=True, show_backcast=False, show_quantiles=True, series_title='Forecast', minutes_per_day=86400//12)
                    
            fig.update_layout(
                font=dict(size=18),
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                ),
                width=1280,
                height=480,
                showlegend=i==j==0,
                legend=dict(
                    yanchor="top",
                    y=0.96,
                    xanchor="left",
                    x=0.01,
                    font_size=18
                ),
                xaxis=dict(
                    title=dict(
                        text='Date & Time',
                        font_size=18
                    )
                ),
                yaxis=dict(
                    title=dict(
                        text='Normalized TWD',
                        font_size=18
                    ),
                    minallowed=-0.05,
                )
            )

            fig.write_image(figures_path.joinpath('forecasts_%s_%i_%i.png' % (args.model, species_name.lower(), series_idx, sample_idx)))