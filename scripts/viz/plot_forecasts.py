
import pandas as pd
import numpy as np

from scripts.viz.utils.data_utils import load_data
from scripts.viz.utils.plot_utils import subplot_forecast_sample_overlay
from scripts.viz.utils.metrics_utils import per_sample_performance

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
arg_parser.add_argument('--samples', type=str, default='main_eval', help='One of [main_eval, fail_eval, random].')

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

    if args.samples == 'main_eval':
        series_samples = {
            160: [41, 406],      # Fagus sylvatica
            166: [126, 1516],    # Pinus sylvestris
            557: [333, 756],     # Picea abies
            1099: [366, 1188],   # Quercus pubescens
            1382: [618, 1180],   # Abies alba
            120: [850, 1187],    # Quercus petraea
        }
    else:
        series_samples = {
            160: [126, 348],     # Fagus sylvatica
            166: [88, 511],      # Pinus sylvestris
            557: [406, 814],     # Picea abies
            1099: [784, 1524],   # Quercus pubescens
            1382: [505, 1409],   # Abies alba
            120: [59, 828],      # Quercus petraea
        }

    n_rows = len(series_samples)
    n_cols = 2

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=False,       # each panel covers a different time window
        shared_yaxes=True,        # share y within each row
        horizontal_spacing=0.05,
        vertical_spacing=0.04,
    )

    for i, (series_idx, sample_list) in enumerate(series_samples.items()):
        mask_ser = (data_ser['series_ids'] == series_idx)
        species_name = '_'.join(metadata_frame.loc[series_idx][['tree_genus', 'tree_species']].to_list())

        # If this option is selected, randomly sample two examples for each series
        if args.samples == 'random':
            sample_list = np.random.choice(np.arange(mask_ser.sum()), size=2, replace=False)

        for j, sample_idx in enumerate(sample_list):
            subplot_forecast_sample_overlay(
                data_ser, mask_ser, sample_idx, fig, i + 1, j + 1, (41, 101, 169),
                show_target=True, show_backcast=False, show_quantiles=True,
                series_title='Forecast', minutes_per_day=86400//12
            )

            mae, medianae, mase, r2 = per_sample_performance(data_ser, mask_ser, sample_idx)

            # Axis index of this subplot (plotly numbers them row-major: 1, 2, 3, ...)
            ax_n = i * n_cols + j + 1
            ax_suffix = '' if ax_n == 1 else str(ax_n)

            # MASE label in the top-left corner of this subplot
            fig.add_annotation(
                xref=f"x{ax_suffix} domain",
                yref=f"y{ax_suffix} domain",
                x=0.01, y=0.95,
                xanchor="left", yanchor="top",
                text="MASE=%.4f" % mase,
                showarrow=False,
                font=dict(size=32, color="black"),
            )

        # Species label on the right side of the row
        ax_right = i * n_cols + n_cols                      # axis index of the last column in this row
        ax_right_suffix = '' if ax_right == 1 else str(ax_right)
        fig.add_annotation(
            text=species_name.replace('_', ' '),
            xref='paper',
            yref=f"y{ax_right_suffix} domain",              # vertically centred on this row
            x=1.01, y=0.5,
            xanchor='left', yanchor='middle',
            textangle=90,
            showarrow=False,
            font=dict(size=32),
        )

    # Keep only one legend entry per trace name (each subplot re-adds the same traces)
    seen = set()
    for tr in fig.data:
        if tr.showlegend is False:
            continue
        key = tr.legendgroup or tr.name
        if key in seen:
            tr.showlegend = False
        else:
            seen.add(key)

    # Same lower bound on every y-axis
    fig.update_yaxes(minallowed=-0.05)

    # One shared y-axis label for the whole figure
    fig.add_annotation(
        text='Normalized TWD',
        xref='paper', yref='paper',
        x=-0.035, y=0.5,
        xanchor='center', yanchor='middle',
        textangle=-90,
        showarrow=False,
        font=dict(size=32),
    )

    fig.update_layout(
        title=dict(
            text='Date & Time',
            y=0.005,
            x=0.5,
            xanchor='center',
            yanchor='bottom',
            font_size=32
        ),
        margin=dict(l=130, r=110, t=120, b=120),
        legend=dict(
            orientation="h",
            entrywidth=200,
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            font_size=32
        ),
        font=dict(size=32),
        width=2560,
        height=440 * n_rows + 240,
        showlegend=True,
    )

    fig.write_image(figures_path.joinpath('%s_all.png' % args.model))