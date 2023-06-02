from __future__ import annotations

from pyroclastmpm.pyroclastmpm_pybind import set_global_step as pyro_set_global_step

from .pyroclastmpm_pybind import CSV, GTFL, OBJ, VTK
from .pyroclastmpm_pybind import (
    set_global_output_directory as pyro_set_global_output_directory,
)
from .pyroclastmpm_pybind import (
    set_global_shapefunction as pyro_set_global_shapefunction,
)
from .pyroclastmpm_pybind import set_global_timestep as pyro_set_global_timestep
from .pyroclastmpm_pybind import set_globals as pyro_set_globals


def set_global_timestep(dt: float):
    """
    Set the simulation timestep in global memory.
    It must be set before any Pyroclast objects are called.
    Args:
        dt (float): simulation timestep
    """
    pyro_set_global_timestep(dt)


def set_global_shapefunction(dimension: int, shape_function):
    """
    Set the simulation shape function in global memory.
    It must be set before any Pyroclast objects are called.
    (warning use set_globals instead)
    """
    pyro_set_global_shapefunction(dimension, shape_function)


def set_global_output_directory(output_directory: str):
    """Sets the output folder


    Args:
        output_directory (str): output directory path
    """
    pyro_set_global_output_directory(output_directory)


def set_global_step(step: int):
    """Sets the output folder

    Args:
        step (int): simulation step
    """
    pyro_set_global_step(step)


def set_globals(
    dt: float, particles_per_cell: int, shape_function, output_directory: str
):
    """Sets the output folder

    Args:
        dt (float): simulation timestep
        particles_per_cell (int): number of particles per cell (initial state)
        shape_function (_type_): shape function
        output_directory (str):  output directory path
    """
    pyro_set_globals(dt, particles_per_cell, shape_function, output_directory)
