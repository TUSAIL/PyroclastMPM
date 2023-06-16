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

import numpy as np

from . import pyroclastmpm_pybind as MPM


class ParticlesContainer(MPM.ParticlesContainer):
    """Particles container class"""

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray = None,
        colors: t.List[int] = None,
        is_rigid: t.List[bool] = None,
        stresses: np.ndarray = None,
        masses: np.ndarray = None,
        volumes: np.ndarray = None,
        output_formats: t.List[t.Type[MPM.OutputFormat]] = None,
    ):
        """Initialize Particles Container

        Args:
            positions (np.ndarray): Coordinates of the particles of
                                    shape (N, D) where N is the number
                                    of particles and D is the dimension
                                    of the problem
            velocities (np.ndarray, optional): Velocities of the particles
                                    of the same shape as positions.
                                    Defaults to None.
            colors (t.List[int], optional): Colors or material type of the
                                    particle of shape (N). Defaults to None.
            is_rigid (t.List[bool], optional): Mask if particles are rigid
                                    or not. Defaults to None.
            stresses (np.ndarray, optional): Initial stress of particles of
                                    shape (N,D,D). Defaults to None.
            masses (np.ndarray, optional): Initial mass of particles of
                                    shape (N). Defaults to None.
            volumes (np.ndarray, optional): Initial volume of particles
                                    of shape (N). Defaults to None.
            output_formats (t.List[str], optional):
                                    List of output formats
                                    ("vtk", "csv", "obj")
                                    Defaults to None.
        """
        if output_formats is None:
            output_formats = []
        if colors is None:
            colors = []
        if is_rigid is None:
            is_rigid = []
        if velocities is None:
            velocities = []
        if stresses is None:
            stresses = []
        if masses is None:
            masses = []
        if volumes is None:
            volumes = []

        out_fmt = []
        for fmt in output_formats:
            if fmt == "vtk":
                out_fmt.append(MPM.VTK)
            elif fmt == "csv":
                out_fmt.append(MPM.CSV)
            elif fmt == "obj":
                out_fmt.append(MPM.OBJ)
            else:
                raise ValueError("Unknown output format: {}".format(fmt))

        super(ParticlesContainer, self).__init__(
            positions=positions,
            velocities=velocities,
            colors=colors,
            is_rigid=is_rigid,
            stresses=stresses,
            masses=masses,
            volumes=volumes,
            output_formats=out_fmt,
        )
