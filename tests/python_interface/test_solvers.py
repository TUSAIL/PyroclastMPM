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

import numpy as np
import pytest
from pyroclastmpm import (
    MUSL,
    TLMPM,
    USL,
    BoundaryCondition,
    Material,
    NodesContainer,
    ParticlesContainer,
    global_dimension,
)

# Functions to test
# [x] USL create ( with and without boundary conditions and materials )
# [x] MUSL create ( with and without boundary conditions and materials )
# [x] TLMPM create ( with and without boundary conditions and materials )
# [ ] USL update
# [ ] MUSL update
# [ ] TLMPM update


@pytest.mark.parametrize("solver_type", [("usl"), ("musl"), ("tlmpm")])
def test_create_solver(solver_type):
    if solver_type == "usl":
        solverclass = USL
    elif solver_type == "musl":
        solverclass = MUSL
    elif solver_type == "tlmpm":
        solverclass = TLMPM
    else:
        assert False, "Invalid solver type"

    if global_dimension == 1:
        pos = np.array([0])
        node_start = np.array([0])
        node_end = np.array([1])
    elif global_dimension == 2:
        pos = np.array([0, 1])
        node_start = np.array([0, 0])
        node_end = np.array([1, 1])
    elif global_dimension == 3:
        pos = np.array([0, 1, 2])
        node_start = np.array([0, 0, 0])
        node_end = np.array([0.4, 0.4, 0.4])

    nodes = NodesContainer(node_start, node_end, node_spacing=0.2)

    particles = ParticlesContainer(positions=[pos])

    # Create a solver
    solver = solverclass(particles, nodes)

    assert isinstance(solver, solverclass)

    dummy_material = Material()
    dummy_boundarycondition = BoundaryCondition()

    # Create a solver
    solver = solverclass(
        particles,
        nodes,
        materials=[dummy_material],
        boundaryconditions=[dummy_boundarycondition],
        alpha=0.5,
    )
    assert isinstance(solver, solverclass)
