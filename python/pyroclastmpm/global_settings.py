from __future__ import annotations

from .pyroclastmpm_pybind import set_globals as pyro_set_globals
from .pyroclastmpm_pybind import set_global_timestep as pyro_set_global_timestep
from .pyroclastmpm_pybind import (
    set_global_shapefunction as pyro_set_global_shapefunction,
)
from .pyroclastmpm_pybind import (
    set_global_output_directory as pyro_set_global_output_directory,
)

from .pyroclastmpm_pybind import VTK, CSV, OBJ

import typing as t

def set_global_timestep(dt: float):
    """Set the simulation timestep in global memory.

    It must be set before any Pyroclast objects are called.

    :param dt: input simulation timestep
    """
    pyro_set_global_timestep(dt)


def set_global_shapefunction(
    dimension: int,
    shape_function
):
    """Set the simulation shape function in global memory.

    It must be set before any Pyroclast objects are called.

    :param shape_function: input simulation shape function
    """

    pyro_set_global_shapefunction(dimension, shape_function)


def set_global_output_directory(
    output_directory: str,
    out_type: t.Union[
        t.Type["VTK"],
        t.Type["CSV"],
        t.Type["OBJ"],
    ]):
    """Sets the output folder

    :param output_directory: input simulation shape function
    """

    pyro_set_global_output_directory(output_directory, out_type)


def set_globals(
    dt: float,
    particles_per_cell: int,
    shape_function,
    output_directory: str
):
    """Sets the output folder

    :param output_directory: input simulation shape function
    """

    pyro_set_globals(dt, particles_per_cell, shape_function, output_directory)
