from __future__ import annotations

import pandas as pd
import numpy as np


def plotly2d(
    df: pd.DataFrame,
    range_x: np.array,
    range_y: np.array,
    x: str = "X",
    y: str = "Y",
    color: str | None = None,
    range_c=None,
    animation_frame: str = "time",
    height: int = 600,
    width: int = 600,
    markersize: int = 3,
):
    """plot a 2D plotly animated graph

    :param df: input dataframe
    :param range_x: domain of x-axis [min,max]
    :param range_y: domain of y-axis [min,max]
    :param x: column in dataframe to be the x-axis, defaults to "X"
    :param y: column in dataframe to be y-axis, defaults to "Y"
    :param color: column in dataframe to be the hue, defaults to None
    :param range_c: data range of hue [min,max], defaults to None
    :param animation_frame: column in dataframe to be the time axis, defaults to "time"
    :param height: height of plotted figure, defaults to 600
    :param width: width of plotted figure, defaults to 600
    :param markersize: point size, defaults to 3
    """
    import plotly.express as px

    fig = px.scatter(
        df,
        x=x,
        y=y,
        animation_frame=animation_frame,
        color=color,
        range_x=range_x,
        range_y=range_y,
        height=height,
        width=width,
        template="plotly_white",
        range_color=range_c,
    )

    fig.update_traces(
        marker={
            "size": markersize,
        },
    )

    fig.update(layout_coloraxis_showscale=True)

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1

    fig.update_xaxes(visible=False)

    fig.update_yaxes(visible=False)

    fig.show()
