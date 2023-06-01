from __future__ import annotations

from .pyroclastmpm_pybind import get_bounds as pyro_get_bounds
from .pyroclastmpm_pybind import grid_points_in_volume as pyro_grid_points_in_volume
from .pyroclastmpm_pybind import grid_points_on_surface as pyro_grid_points_on_surface
from .pyroclastmpm_pybind import set_device as pyro_set_device
from .pyroclastmpm_pybind import (
    uniform_random_points_in_volume as pyro_uniform_random_points_in_volume,
)


def uniform_random_points_in_volume(stl_filename: str, num_points: int):
    return pyro_uniform_random_points_in_volume(stl_filename, num_points)


def grid_points_in_volume(stl_filename: str, cell_size: float, point_per_cell):
    return pyro_grid_points_in_volume(stl_filename, cell_size, point_per_cell)


def grid_points_on_surface(stl_filename: str, cell_size: float, point_per_cell):
    return pyro_grid_points_on_surface(stl_filename, cell_size, point_per_cell)


def get_bounds(stl_filename: str):
    return pyro_get_bounds(stl_filename)


def set_device(device_id: id):
    return pyro_set_device(device_id)
