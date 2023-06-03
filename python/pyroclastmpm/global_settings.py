from __future__ import annotations

import typing as t

from . import pyroclastmpm_pybind as MPM


def set_global_timestep(dt: float):
    """
    Set the simulation timestep in global memory.
    It must be set before any Pyroclast objects are called.
    Args:
        dt (float): simulation timestep
    """
    MPM.set_global_timestep(dt)


def set_global_shapefunction(shape_function: t.Type[MPM.ShapeFunction]):
    """
    Set the simulation shape function in global memory.
    It must be set before any Pyroclast objects are called.
    Args:
        dt (ShapeFunction): ShapeFunction object e.g. CubicShapeFunction,
        LinearShapeFunction
    """
    MPM.set_global_shapefunction(shape_function)


def set_global_output_directory(output_directory: str):
    """Sets the output folder
    Args:
        output_directory (str): output directory path
    """
    MPM.set_global_output_directory(output_directory)


def set_global_step(step: int):
    """Sets the output folder
    Args:
        step (int): simulation step
    """
    MPM.set_global_step(step)


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
    MPM.set_globals(dt, particles_per_cell, shape_function, output_directory)
