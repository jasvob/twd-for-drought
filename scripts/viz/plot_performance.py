import numpy as np
import pandas as pd

from scripts.viz.utils.data_utils import load_data
from scripts.viz.utils.metrics_utils import overall_performance, species_stats, per_species_performance, per_species_performance_paper

import plotly.graph_objects as go

import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Plot Performance',
    description='Create performance plots for various models.',
)

arg_parser.add_argument('--data_path', type=str, help='Path to the dataset folder.')   
arg_parser.add_argument('--output_path', type=str, help='Path to where the outputs should be stored.')
arg_parser.add_argument('--model', type=str, default='nhits', help='One of [nhits, nhits_noswp, tft, tft_noswp, rf, xgb].')


if __name__ == "__main__":
    args = arg_parser.parse_args()

    # Create output folders if they don't exist
    figures_path = Path(args.output_path).joinpath('figures')
    figures_path.mkdir(parents=True, exist_ok=True)
    tables_path = Path(args.output_path).joinpath('tables')
    tables_path.mkdir(parents=True, exist_ok=True)

    # Load tree metadata
    metadata_frame = pd.read_parquet(Path(args.data_path).joinpath('filt_metadata.parquet'))
    metadata_frame = metadata_frame.set_index('series_id')

    # Load model evaluation data
    data_ser = load_data(Path(args.output_path).joinpath('eval'), args.model)

    # Define which series IDs to consider in this evaluation
    filter_sids = [
        172, 1099, 1351, 1369, # Quercus pubescens
        120, 853, # Quercus petraea:
        1, 3, 166, 696, 1208, 1210, 1270, 1276, 1365, 1396, # Pinus sylvestris
        21, 27, 557, 686, 1180, 1206, 1268, 1381, # Picea Abies
        18, 156, 160, 687, 849, 855, 1094, 1169, 1218, 1385, # Fagus sylvatica
        28, 693, 1163, 1382 # Abies alba
    ]

    # Prepare a text file to write to
    out_file = open(tables_path.joinpath('performance_%s.txt' % args.model), 'w')

    # Calculate overall performance
    results = {}
    results['all_ser'] = overall_performance(data_ser, filter_sids, args.model.upper(), file=out_file)
    print('\n', file=out_file)

    # Display species statistics
    species_stats(data_ser, metadata_frame, filter_sids, args.model.upper(), file=out_file)
    print('\n', file=out_file)

    # Calcualte performance per-species
    results['per_species_ser'] = per_species_performance(data_ser, metadata_frame, filter_sids, args.model.upper(), file=out_file)
    print('\n', file=out_file)

    # Plot performance per-species in paper table ready format
    per_species_performance_paper(data_ser, metadata_frame, filter_sids, args.model.upper(), file=out_file)
    print('\n', file=out_file)

    out_file.close()

    fig = go.Figure()

    error_name = 'MASE'
    yvals_ser = results['per_species_ser']['mase']
    xvals_ser = np.arange(yvals_ser.shape[1])

    for i in range(yvals_ser.shape[0]):
        fig.add_trace(go.Scatter(
            x=xvals_ser,
            y=yvals_ser[i],
            mode='lines+markers',
            line_width=2,
            name='%s' % results['per_species_ser']['spec_name'][i],
        ))


    fig.update_layout(
        xaxis=dict(
            title=dict(
                text='Forecasting Horizon [2h]',
                font_size=18
            )
        ),
        yaxis=dict(
            title=dict(
                text=error_name,
                font_size=18
            )
        ),
        font=dict(size=18),
        width=1280,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font_size=18
        )
    )

    fig.write_image(figures_path.joinpath('performance_per_species_%s.png' % args.model))