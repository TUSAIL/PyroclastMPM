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

from tqdm import tqdm

from . import pyroclastmpm_pybind as MPM
from .boundaryconditions import BoundaryCondition
from .materials import Material
from .nodes import NodesContainer
from .particles import ParticlesContainer

Logo = f" \n \
 \n \
🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️ \n \
🔥️█▀█ █▄█ █▀█ █▀█ █▀▀ █░░ ▄▀█ █▀ ▀█▀ █▀▄▀█ █▀█ █▀▄▀█🔥️ \n \
🔥️█▀▀ ░█░ █▀▄ █▄█ █▄▄ █▄▄ █▀█ ▄█ ░█░ █░▀░█ █▀▀ █░▀░█🔥️ \n \
🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️🔥️ \n \
Running {MPM.global_dimension}D simulation \n \
"


class USL(MPM.USL):
    """
    Update Stress Last Solver

    Example usage:

    .. highlight:: python
    .. code-block:: python

        MPM = Solver(
            particles=particles, # of type ParticlesContainer
            nodes=nodes, # of type NodesContainer
            total_steps=40000,
            output_steps=1000,
            output_start=0,
            boundaryconditions=[gravity], # list of types BoundaryCondition
        )
    """

    #: Total number of steps to finish (interfaced with pybind11)
    total_steps: int

    #: Step when the output start (interfaced with pybind11)
    output_start: int

    #: The nth steps to output (interfaced with pybind11)
    output_steps: int

    def __init__(
        self,
        particles: t.Type["ParticlesContainer"],
        nodes: t.Type["NodesContainer"],
        boundaryconditions: t.List[t.Type["BoundaryCondition"]] = None,
        materials: t.List[t.Type["Material"]] = None,
        alpha=0.99,
        total_steps: int = 0,
        output_steps: int = 0,
        output_start: int = 0,
    ):
        """Initialize Solver

        Args:
            particles (t.Type[&quot;ParticlesContainer&quot;]):
                                    Particles container object
            nodes (t.Type[&quot;NodesContainer&quot;]):
                                    Nodes container object
            boundaryconditions
              (t.List[t.Type[&quot;BoundaryCondition&quot;]], optional):
                                    List of boundary conditions.
                                    Defaults to None.
            materials (_type_, optional): List of materials. Defaults to None.
            alpha (float, optional): Ratio of FLIP/PIC. Defaults to 0.99.
            total_steps (int, optional): Total number of steps to solve.
                                         Defaults to 0.
            output_steps (int, optional): Save files for every i'th step. Defaults to 0.
            output_start (int, optional): The step at which the output starts.
                                         Defaults to 0.
        """
        if boundaryconditions is None:
            boundaryconditions = []
        if materials is None:
            materials = []

        self.total_steps = total_steps
        self.output_steps = output_steps
        self.output_start = output_start

        #: init c++ class from pybind11
        super(USL, self).__init__(
            particles=particles,
            nodes=nodes,
            materials=materials,
            boundaryconditions=boundaryconditions,
            alpha=alpha,
        )

    def solve_nsteps(self, num_steps: int):
        """Solve the simulation for a specified number of steps.

        Args:
            num_steps (int): Number of steps to solve
        """
        super(USL, self).solve_nsteps(num_steps)

    def run(self, callback: t.Callable = None):
        """Run the simulation for the specified number of steps
        Args:
            callback (t.Callable, optional): Callback function called after every i'th
                                            output step.Defaults to None.
        """
        print(Logo)

        self.solve_nsteps(self.output_start)

        for step in tqdm(
            range(self.output_start, self.total_steps, self.output_steps),
            desc="PyroclastMPM ",
            unit=" step",
            colour="green",
            unit_divisor=self.output_steps,
        ):
            self.solve_nsteps(self.output_steps)
            if callback is not None:
                callback()
