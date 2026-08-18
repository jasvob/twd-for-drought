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

def plot_attributions_series(timestamps : np.array, predictions : np.array, attributions : np.array, feature_list : List[str]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(timestamps * 10**9),
            y=predictions,
            mode='lines',
            line_width=3,
            marker_color='green',
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Heatmap(
            z=attributions.transpose(1,0)[:6, :] / attributions[:, :6].abs().max(),
            x=pd.to_datetime(timestamps * 10**9),
            y=-1.5 + np.arange(attributions.shape[1]) * 1.5,
            colorscale=px.colors.sequential.RdBu_r[1:10],
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(
                title=dict(
                    text='Contribution',
                    font_size=18
                ),
                tickvals=[1.0, 0.5, 0.0, -0.5, -1.0],
                ticktext=['pos. high', 'pos. low', 'none', 'neg. low', 'neg. high'],
                x=1.0
            )
        ),
        secondary_y=False
    )

    fig.update_layout(
        width=1200,
        height=400,
        yaxis=dict(
            title=dict(
                text='Input Features',
                font_size=16
            ),
            tickmode='array',
            tickvals=-0.75 + np.array(list(range(len(feature_list)))) * 1.5,
            ticktext=feature_list
        ),
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        font=dict(size=16),
        xaxis=dict(
            title=dict(
                text='Date & Time',
                font_size=16,
            )
        ),
        yaxis2=dict(
            title=dict(
                text='Forecasted TWD',
                font_size=16,
            )
        )
    )

    return fig