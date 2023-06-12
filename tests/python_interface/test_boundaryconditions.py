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
from pyroclastmpm import (
    BodyForce,
    Gravity,
    NodeDomain,
    PlanarDomain,
    RigidParticles,
    global_dimension,
)

# Functions tested
# [x] Gravity initialization (without ramp)
# [x] Gravity initialization ramp (with ramp)
# [x] BodyForce initialization (mode = "force","momentum","Fixed")
# [x] PeriodicWall initialization ("x", "y", "z")
# [x] PlanarDomain initialization
# [x] NodeDomain initialization
# [x] NodeDomain initialization
# [ ] RigidParticles initialization ("positions"
# + "positions","frames","locations","rotations")


def test_planardomain_create():
    if global_dimension == 1:
        axis0_friction = np.array([0])
        axis1_friction = np.array([9.8])
    elif global_dimension == 2:
        axis0_friction = np.array([0, 0])
        axis1_friction = np.array([9.8, 9.8])
    elif global_dimension == 3:
        axis0_friction = np.array([0, 0, 0])
        axis1_friction = np.array([9.8, 9.8, 9.8])

    boundarycondition = PlanarDomain(
        axis0_friction=axis0_friction, axis1_friction=axis1_friction
    )

    assert isinstance(boundarycondition, PlanarDomain)


def test_nodedomain_create():
    if global_dimension == 1:
        axis0_mode = np.array([0])
        axis1_mode = np.array([1])
    elif global_dimension == 2:
        axis0_mode = np.array([1, 0])
        axis1_mode = np.array([2, 0])
    elif global_dimension == 3:
        axis0_mode = np.array([0, 1, 0])
        axis1_mode = np.array([1, 2, 1])

    boundarycondition = NodeDomain(
        axis0_mode=axis0_mode, axis1_mode=axis1_mode
    )

    assert isinstance(boundarycondition, NodeDomain)


def test_gravity_create():
    if global_dimension == 1:
        gravity_start = np.array([0])
        gravity_end = np.array([-9.8])
    elif global_dimension == 2:
        gravity_start = np.array([0, 0])
        gravity_end = np.array([-9.8, -9.8])
    elif global_dimension == 3:
        gravity_start = np.array([0, 0, 0])
        gravity_end = np.array([-9.8, -9.8, -9.8])

    boundarycondition = Gravity(gravity=gravity_start)

    assert isinstance(boundarycondition, Gravity)

    boundarycondition = Gravity(
        gravity=gravity_start,
        is_ramp=True,
        ramp_step=10,
        gravity_end=gravity_end,
    )

    assert isinstance(boundarycondition, Gravity)


def test_bodyforce_create():
    mask = np.array([False, True])
    if global_dimension == 1:
        values = np.array([[2.0], [3.0]])
    elif global_dimension == 2:
        values = np.array([[2.0, 7.0], [3.0, 5.0]])
    elif global_dimension == 3:
        values = np.array([[2.0, 7.0, 11], [3.0, 5.0, 10]])

    boundarycondition = BodyForce(mode="forces", values=values, mask=mask)

    assert isinstance(boundarycondition, BodyForce)

    assert boundarycondition.mode_id == 0

    boundarycondition = BodyForce(mode="moments", values=values, mask=mask)

    assert isinstance(boundarycondition, BodyForce)

    assert boundarycondition.mode_id == 1

    boundarycondition = BodyForce(mode="fixed", values=values, mask=mask)

    assert isinstance(boundarycondition, BodyForce)

    assert boundarycondition.mode_id == 2


def test_create_rigidparticles():
    if global_dimension == 1:
        pos = np.array([[2.0], [3.0]])
        frames = np.array([0, 1])
        locations = np.array([[0.0], [1.0]])
        rotations = np.array([[0.0], [0.0]])
    elif global_dimension == 2:
        pos = np.array([[2.0, 7.0], [3.0, 5.0]])
        frames = np.array([0, 1])
        locations = np.array([[0.0, 2], [1.0, 3]])
        rotations = np.array([[0.0, 1], [0.0, 3]])
    elif global_dimension == 3:
        pos = np.array([[2.0, 7.0, 11], [3.0, 5.0, 10]])
        frames = np.array([0, 1])
        locations = np.array([[0.0, 2, 3], [1.0, 3, 4]])
        rotations = np.array([[0.0, 1, 0], [0.0, 3, 1]])

    boundarycondition = RigidParticles(pos)

    assert isinstance(boundarycondition, RigidParticles)

    boundarycondition = RigidParticles(pos, frames, locations, rotations)

    assert isinstance(boundarycondition, RigidParticles)
