import numpy as np


def create_circle(
    center: np.array, radius: float, cell_size: float, ppc_1d: int = 1
):
    start, end = center - radius, center + radius
    spacing = cell_size / ppc_1d
    tol = +0.00005  # prevents points
    x = np.arange(start[0], end[0] + spacing, spacing) + 0.5 * spacing
    y = np.arange(start[1], end[1] + spacing, spacing) + 0.5 * spacing
    xv, yv = np.meshgrid(x, y)
    grid_coords = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(
        np.float64
    )
    circle_mask = (grid_coords[:, 0] - center[0]) ** 2 + (
        grid_coords[:, 1] - center[1]
    ) ** 2 < radius**2 + tol
    return grid_coords[circle_mask]
