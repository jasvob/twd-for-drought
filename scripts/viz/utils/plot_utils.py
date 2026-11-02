import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from typing import Tuple, List

def plot_series(data_dict, sid, fig : go.Figure, forecast_color : Tuple[int, int, int], show_target : bool = True, show_quantiles : bool = False, series_title : str = 'Forecast'):
    mask = (data_dict['series_ids'] == sid)

    forecast_idx = 0
    quant_idx = data_dict['forecasts'].shape[-1] // 2
    xvalsall = pd.to_datetime(data_dict['tstamps'][mask] * 10**9).strftime('%y-%m-%d %H:%M:%S')

    if show_target:
        fig.add_trace(go.Scatter(
            x=xvalsall,
            y=data_dict['fcast_tgts'][mask, forecast_idx, 0],
            mode='lines',
            marker_color='rgba(32, 138, 73, 1.0)',
            line_width=4,
            name='Target',
        ))
    fig.add_trace(go.Scatter(
        x=xvalsall,
        y=data_dict['forecasts'][mask, forecast_idx, quant_idx],
        mode='lines',
        marker_color='rgba(%i, %i, %i, 1.0)' % forecast_color,
        line_width=4,
        name=series_title,
        text=np.arange(xvalsall.shape[0])
    ))
    
    if show_quantiles:
        if data_dict['forecasts'].shape[2] > 1:
            fig.add_trace(go.Scatter(
                x=xvalsall,
                y=data_dict['forecasts'][mask, forecast_idx, 0],
                mode='lines',
                marker_color='rgba(%i, %i, %i, 0.5)' % forecast_color,
                name=series_title,
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=xvalsall,
                y=data_dict['forecasts'][mask, forecast_idx, 2 * quant_idx],
                mode='lines',
                marker_color='rgba(%i, %i, %i, 0.5)' % forecast_color,
                name=series_title,
                fillcolor='rgba(%i, %i, %i, 0.25)' % forecast_color,
                fill='tonexty',
                showlegend=False
            ))


def plot_forecast_sample_overlay(data_dict, data_mask, sample_idx : int, fig : go.Figure, forecast_color : Tuple[int, int, int], show_target : bool = True, show_backcast : bool = False, show_quantiles : bool = False, series_title : str = 'Forecast', minutes_per_day : int = 86400):
    quant_idx = data_dict['forecasts'].shape[-1] // 2

    xvalsb = [data_dict['tstamps'][data_mask][sample_idx] - i * minutes_per_day for i in range(data_dict['bcast_tgts'].shape[1], 0, -1)]
    xvalsf = [data_dict['tstamps'][data_mask][sample_idx] + i * minutes_per_day for i in range(data_dict['forecasts'].shape[1])]
    xvalsall = xvalsb + xvalsf

    xvalsb = pd.to_datetime(np.array(xvalsb) * 10**9)
    xvalsf = pd.to_datetime(np.array(xvalsf) * 10**9)
    xvalsall = pd.to_datetime(np.array(xvalsall) * 10**9)

    if show_target:
        fig.add_trace(go.Scatter(
            x=xvalsb,
            y=data_dict['bcast_tgts'][data_mask, :, 0][sample_idx, :],
            mode='lines',
            marker_color='rgba(32, 138, 73, 1.0)',
            name='History'
        ))
        fig.add_trace(go.Scatter(
            x=xvalsf,
            y=data_dict['fcast_tgts'][data_mask, :, 0][sample_idx, :],
            mode='lines',
            marker_color='rgba(32, 138, 73, 1.0)',
            line_width=4,
            name='Target',
        ))

    if show_backcast:
        fig.add_trace(go.Scatter(
            x=xvalsb,
            y=data_dict['bcast_tgts'][data_mask, :, 0][sample_idx, :] - data_dict['residuals'][data_mask, :, 0][sample_idx, :],
            mode='lines',
            marker_color='rgba(139, 0, 0, 1.0)',
            name='Residuals'
        ))

    fig.add_trace(go.Scatter(
        x=xvalsf,
        y=data_dict['forecasts'][data_mask, :, quant_idx][sample_idx, :],
        mode='lines',
        marker_color='rgba(%i, %i, %i, 1.0)' % forecast_color,
        line_width=4,
        name=series_title,
        text=np.arange(xvalsf.shape[0])
    ))
    
    if show_quantiles:
        if data_dict['forecasts'].shape[2] > 1:
            fig.add_trace(go.Scatter(
                x=xvalsf,
                y=data_dict['forecasts'][data_mask, :, 0][sample_idx, :],
                mode='lines',
                marker_color='rgba(%i, %i, %i, 0.4)' % forecast_color,
                name=series_title,
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=xvalsf,
                y=data_dict['forecasts'][data_mask, :, 2 * quant_idx][sample_idx, :],
                mode='lines',
                marker_color='rgba(%i, %i, %i, 0.4)' % forecast_color,
                name=series_title,
                fillcolor='rgba(%i, %i, %i, 0.2)' % forecast_color,
                fill='tonexty',
                showlegend=False
            ))



def subplot_forecast_sample_overlay(data_dict, data_mask, sample_idx : int, fig : go.Figure, row : int, col : int, forecast_color : Tuple[int, int, int], show_target : bool = True, show_backcast : bool = False, show_quantiles : bool = False, series_title : str = 'Forecast', minutes_per_day : int = 86400):
    quant_idx = data_dict['forecasts'].shape[-1] // 2

    xvalsb = [data_dict['tstamps'][data_mask][sample_idx] - i * minutes_per_day for i in range(data_dict['bcast_tgts'].shape[1], 0, -1)]
    xvalsf = [data_dict['tstamps'][data_mask][sample_idx] + i * minutes_per_day for i in range(data_dict['forecasts'].shape[1])]
    xvalsall = xvalsb + xvalsf

    xvalsb = pd.to_datetime(np.array(xvalsb) * 10**9)
    xvalsf = pd.to_datetime(np.array(xvalsf) * 10**9)
    xvalsall = pd.to_datetime(np.array(xvalsall) * 10**9)

    if show_target:
        fig.add_trace(go.Scatter(
            x=xvalsb,
            y=data_dict['bcast_tgts'][data_mask, :, 0][sample_idx, :],
            mode='lines',
            marker_color='rgba(32, 138, 73, 1.0)',
            name='History',
            showlegend=row==1 and col==1
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=xvalsf,
            y=data_dict['fcast_tgts'][data_mask, :, 0][sample_idx, :],
            mode='lines',
            marker_color='rgba(32, 138, 73, 1.0)',
            line_width=4,
            name='Target',
            showlegend=row==1 and col==1
        ), row=row, col=col)

    if show_backcast:
        fig.add_trace(go.Scatter(
            x=xvalsb,
            y=data_dict['bcast_tgts'][data_mask, :, 0][sample_idx, :] - data_dict['residuals'][data_mask, :, 0][sample_idx, :],
            mode='lines',
            marker_color='rgba(139, 0, 0, 1.0)',
            name='Residuals',
            showlegend=row==1 and col==1
        ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=xvalsf,
        y=data_dict['forecasts'][data_mask, :, quant_idx][sample_idx, :],
        mode='lines',
        marker_color='rgba(%i, %i, %i, 1.0)' % forecast_color,
        line_width=4,
        name=series_title,
        text=np.arange(xvalsf.shape[0]),
        showlegend=row==1 and col==1
    ), row=row, col=col)
    
    if show_quantiles:
        if data_dict['forecasts'].shape[2] > 1:
            fig.add_trace(go.Scatter(
                x=xvalsf,
                y=data_dict['forecasts'][data_mask, :, 0][sample_idx, :],
                mode='lines',
                marker_color='rgba(%i, %i, %i, 0.4)' % forecast_color,
                name=series_title,
                showlegend=False
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=xvalsf,
                y=data_dict['forecasts'][data_mask, :, 2 * quant_idx][sample_idx, :],
                mode='lines',
                marker_color='rgba(%i, %i, %i, 0.4)' % forecast_color,
                name=series_title,
                fillcolor='rgba(%i, %i, %i, 0.2)' % forecast_color,
                fill='tonexty',
                showlegend=False
            ), row=row, col=col)


def plot_attributions_histogram(fig : go.Figure, attrs_all : np.array, feat_names : List[str]):
    fig = ff.create_distplot(attrs_all, list(feat_names), bin_size=0.2, show_hist=False)
    fig.update_layout(
        width=1600, 
        height=1200, 
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def plot_attributions_boxplot(fig : go.Figure, attrs_all : np.array, feat_names : List[str]):
    # Use x instead of y argument for horizontal plot
    for i in range(len(attrs_all)):
        fig.add_trace(go.Box(
            x=attrs_all[i], 
            name=feat_names[i],
        ))

    fig.update_layout(
        width=1200, 
        height=900, 
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        showlegend=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )


# Colour-blind-safe bar colours, one per feature (in feature_list order).
inp_names = ['prec', 'rh', 'temp', 'swp', 'vpd', 'rad']
inp_colors = ['rgba(0, 168, 196, 1.0)', 'rgba(222, 26, 82, 1.0)', 'rgba(193, 187, 0, 1.0)', 'rgba(181, 61, 181, 1.0)', 'rgba(212, 150, 0, 1.0)', 'rgba(15, 130, 222, 1.0)']
     
 
def plot_attributions_series(timestamps_list : List[np.array],
                                predictions_list : List[np.array],
                                attributions_list : List[np.array],
                                feature_list : List[str],
                                subplot_titles : List[str] = None,
                                bar_freq : str = '6h',
                                shared_xaxes : bool = False,
                                row_height : int = 350) -> go.Figure:
    n_rows = len(timestamps_list)

    fig = make_subplots(
        rows=n_rows, cols=1,
        specs=[[{"secondary_y": True}] for _ in range(n_rows)],
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        vertical_spacing=0.08,
    )

    for r, (timestamps, predictions, attributions) in enumerate(
            zip(timestamps_list, predictions_list, attributions_list), start=1):

        show_legend = (r == 1)   # legend entries only once; legendgroup links them across rows

        x = pd.to_datetime(timestamps * 10**9)

        # works for torch tensors and numpy arrays
        to_np = lambda a: a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
        attr = to_np(attributions)
        preds = to_np(predictions)

        n_features = attr.shape[1] - 1
        attr = attr[:, :n_features] / np.abs(attr[:, :n_features]).max()   # same scaling as before (per series)

        # Average contributions into wider bins for the bars; TWD line keeps full resolution
        attr_df = pd.DataFrame(attr, index=x)
        if bar_freq is not None:
            attr_df = attr_df.resample(bar_freq).mean().dropna()
            attr_df.index = attr_df.index + pd.Timedelta(bar_freq) / 2   # centre bars on their bin
        attr = attr_df.to_numpy()

        # Contribution bars (primary axis, placed on the right)
        for i in range(n_features):
            fig.add_trace(
                go.Bar(
                    x=attr_df.index,
                    y=attr[:, i],
                    name=feature_list[i],
                    marker_color=inp_colors[i],
                    marker_line_width=0.5,
                    legendgroup=feature_list[i],
                    showlegend=show_legend,
                ),
                row=r, col=1, secondary_y=False
            )

        # Forecasted TWD (secondary axis, placed on the left)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=preds,
                mode='lines',
                line_width=3,
                marker_color='rgba(20, 40, 90, 1.0)',
                name='Forecasted TWD',
                legendgroup='Forecasted TWD',
                showlegend=show_legend,
            ),
            row=r, col=1, secondary_y=True
        )

        # Centre the contribution axis on 0
        stack_max = max(np.clip(attr, 0, None).sum(axis=1).max(),
                        -np.clip(attr, None, 0).sum(axis=1).min())

        fig.update_yaxes(
            title=dict(text='Contribution', font_size=16),
            side='right',
            zeroline=True, zerolinecolor='grey', zerolinewidth=1,
            range=[-1.05 * stack_max, 1.05 * stack_max],
            row=r, col=1, secondary_y=False,
        )
        fig.update_yaxes(
            title=dict(text='Forecasted TWD', font_size=16),
            side='left',
            showgrid=False,
            tickmode='auto',
            row=r, col=1, secondary_y=True,
        )

    # x-axis title only on the bottom row
    fig.update_xaxes(title=dict(text='Date & Time', font_size=16), row=n_rows, col=1)

    fig.update_layout(
        width=1200,
        height=row_height * n_rows,
        barmode='relative',          # positives stack up from 0, negatives down
        bargap=0,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=16),
    )

    return fig