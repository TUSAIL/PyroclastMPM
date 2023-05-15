import numpy as np
from __future__ import annotations

def create_line(start, end,padding_pos, gap):
    coords  = np.arange(start, end + gap, gap).reshape((-1,1))
    # make (N,1) -> (N,3) required for solver
    padding = np.zeros((coords.shape[0], 1)) + padding_pos
    coords_padded = np.hstack([coords, padding,padding])
    return coords_padded

def create_circle(start, end,padding_pos, gap):
    grid_start = np.array(start)
    grid_end = np.array(end)
    center = (grid_end + grid_start) / 2
    radius = np.mean(grid_end - grid_start) / 2
    height = np.arange(grid_start[0], grid_end[0] + gap, gap)
    width = np.arange(grid_start[1], grid_end[1] + gap, gap)
    xv, yv = np.meshgrid(width, height)
    coords = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)
    circle_mask = (coords[:, 0] - center[0]) ** 2 + (
        coords[:, 1] - center[1]
    ) ** 2 < radius**2
    
    # make (N,1) -> (N,3) required for solver
    padding = np.zeros((coords[circle_mask].shape[0], 1)) + padding_pos
    coords_padded = np.hstack([coords[circle_mask], padding])
    return coords_padded

def create_square(start, end, gap):
    grid_start = np.array(start)
    grid_end = np.array(end)
    center = (grid_end + grid_start) / 2
    radius = np.mean(grid_end - grid_start) / 2
    height = np.arange(grid_start[0], grid_end[0] + gap, gap)
    width = np.arange(grid_start[1], grid_end[1] + gap, gap)
    xv, yv = np.meshgrid(width, height)
    coords = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

    return coords

def create_sphere(start, end, gap):
    grid_start = np.array(start)
    grid_end = np.array(end)
    center = (grid_end + grid_start) / 2
    radius = np.mean(grid_end - grid_start) / 2
    height = np.arange(grid_start[0], grid_end[0] + gap, gap)
    width = np.arange(grid_start[1], grid_end[1] + gap, gap)
    depth = np.arange(grid_start[1], grid_end[1] + gap, gap)
    xv, yv, zv = np.meshgrid(height, width, depth)
    coords = np.array(list(zip(xv.flatten(), yv.flatten(), zv.flatten()))).astype(
        np.float64
    )
    circle_mask = (coords[:, 0] - center[0]) ** 2 + (
        coords[:, 1] - center[1]
    ) ** 2 + (
        coords[:, 2] - center[2]
    ) ** 2 < radius**2
    return coords[circle_mask]


    # circle_mask = (coords[:, 0] - center[0]) ** 2 + (
    #     coords[:, 1] - center[1]
    # ) ** 2 + (
    #     coords[:, 2] - center[2]
    # ) ** 2 < radius**2
    return coords