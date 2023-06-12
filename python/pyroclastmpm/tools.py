# BSD 3-Clause License
# Copyright (c) 2023, Retief Lubbe
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this
#  list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
