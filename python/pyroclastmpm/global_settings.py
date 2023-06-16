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


def set_global_timestep(dt: float):
    """
    Set the simulation timestep in global memory.
    It must be set before any Pyroclast objects are called.
    Args:
        dt (float): simulation timestep
    """
    MPM.set_global_timestep(dt)


def set_global_shapefunction(shape_function):
    """Sets the global shape function type

    Args:
        shape_function (str): shape function type ("cubic" or "linear")
    """
    shp_type = None
    if shape_function == "cubic":
        shp_type = MPM.CubicShapeFunction
    elif shape_function == "linear":
        shp_type = MPM.LinearShapeFunction
    else:
        ValueError("Unknown shape function type: {}".format(shape_function))
    MPM.set_global_shapefunction(shp_type)


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
    dt: float,
    particles_per_cell: int,
    shape_function: str,
    output_directory: str,
):
    """Sets the output folder

    Args:
        dt (float): simulation timestep
        particles_per_cell (int): number of particles per cell (initial state)
        shape_function (str): shape function type ("cubic" or "linear")
        output_directory (str):  output directory path
    """

    shp_type = None
    if shape_function == "cubic":
        shp_type = MPM.CubicShapeFunction
    elif shape_function == "linear":
        shp_type = MPM.LinearShapeFunction
    else:
        ValueError("Unknown shape function type: {}".format(shape_function))

    MPM.set_globals(dt, particles_per_cell, shp_type, output_directory)
