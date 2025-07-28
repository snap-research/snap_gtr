import numpy as np
import plotly.graph_objects as go

import torch


def vis_rays(origins, directions, color='lightblue') -> go.Figure:  # type: ignore
    """Visualize camera rays.

    Args:
        camera: Camera to visualize.

    Returns:
        Plotly lines
    """
    origins = origins.view(-1, 3)
    directions = directions.view(-1, 3)

    lines = torch.empty((origins.shape[0] * 2, 3))
    lines[0::2] = origins
    lines[1::2] = origins + directions

    fig = go.Figure(  # type: ignore
        data=go.Scatter3d(  # type: ignore
            x=lines[:, 0],
            y=lines[:, 2],
            z=lines[:, 1],
            marker=dict(
                size=4,
            ),
            line=dict(color=color, width=1),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", showspikes=False),
            yaxis=dict(title="z", showspikes=False),
            zaxis=dict(title="y", showspikes=False),
        ),
        margin=dict(r=0, b=10, l=0, t=10),
        hovermode=False,
    )

    return fig


def vis_points(points, color='lightblue') -> go.Figure:
    points = points.view(-1, 3)

    fig = go.Figure(  # type: ignore
        data=go.Scatter3d(  # type: ignore
            x=points[:, 0],
            y=points[:, 2],
            z=points[:, 1],
            marker=dict(
                size=4,
            ),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", showspikes=False),
            yaxis=dict(title="z", showspikes=False),
            zaxis=dict(title="y", showspikes=False),
        ),
        margin=dict(r=0, b=10, l=0, t=10),
        hovermode=False,
    )

    return fig


def vis_ray_samples(origins, directions, samples, color='lightblue', num_rays=10) -> go.Figure:  # type: ignore
    """Visualize camera rays.

    Args:
        camera: Camera to visualize.

    Returns:
        Plotly lines
    """
    origins = origins.view(-1, 3)
    directions = directions.view(-1, 3)
    num_rays = origins.shape[0]
    samples = samples.view(num_rays, -1, 3)

    lines = torch.empty((origins.shape[0] * 2, 3))
    lines[0::2] = origins
    lines[1::2] = origins + directions

    fig = go.Figure(  # type: ignore
        data=go.Scatter3d(  # type: ignore
            x=lines[:, 0],
            y=lines[:, 2],
            z=lines[:, 1],
            marker=dict(
                size=4,
            ),
            line=dict(color=color, width=1),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", showspikes=False),
            yaxis=dict(title="z", showspikes=False),
            zaxis=dict(title="y", showspikes=False),
        ),
        margin=dict(r=0, b=10, l=0, t=10),
        hovermode=False,
    )

    return fig