from __future__ import annotations

import typing as t

from . import pyroclastmpm_pybind as MPM


def uniform_random_points_in_volume(
    stl_filename: str, num_points: int
) -> t.List[t.Tuple[float, float, float]]:
    """Generate uniform random points within a volume

    Args:
        stl_filename (str): stl file name
        num_points (int): number of points to generate

    Returns:
        t.List[t.Tuple[float, float, float]]: list of coordinates
    """
    return MPM.uniform_random_points_in_volume(stl_filename, num_points)


def grid_points_in_volume(
    stl_filename: str, cell_size: float, point_per_cell
) -> t.List[t.Tuple[float, float, float]]:
    """Generate grid points within a volume

    Args:
        stl_filename (str): stl file name
        cell_size (float): Grid cell size
        point_per_cell (_type_): Number of points per cell

    Returns:
        t.List[t.Tuple[float, float, float]]: list of coordinates
    """
    return MPM.grid_points_in_volume(stl_filename, cell_size, point_per_cell)


def grid_points_on_surface(
    stl_filename: str, cell_size: float, point_per_cell
) -> t.List[t.Tuple[float, float, float]]:
    """Generate grid points on the surface

    Args:
        stl_filename (str): stl file name
        cell_size (float): Grid cell size
        point_per_cell (_type_): Number of points per cell

    Returns:
        t.List[t.Tuple[float, float, float]]: list of coordinates
    """
    return MPM.grid_points_on_surface(stl_filename, cell_size, point_per_cell)


def get_bounds(
    stl_filename: str,
) -> t.Tuple[t.Tuple[float, float, float], t.Tuple[float, float, float]]:
    """Get start and end coordinates of an STL file

    Args:
        stl_filename (str): stl file name

    Returns:
        t.Tuple[t.Tuple[float, float, float], t.Tuple[float, float, float]]:
        Tuple of start and end coordinates
    """
    return MPM.get_bounds(stl_filename)


def set_device(device_id: id):
    """Set the GPU device to use

    Args:
        device_id (id): _description_

    Returns:
        _type_: _description_
    """
    return MPM.set_device(device_id)
