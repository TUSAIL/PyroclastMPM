from pyroclastmpm import (
    Gravity,
    BodyForce,
    RigidParticles,
    PlanarDomain,
    NodeDomain,
    global_dimension
)

import numpy as np

# Functions tested
# [x] Gravity initialization (without ramp)
# [x] Gravity initialization ramp (with ramp)
# [x] BodyForce initialization (mode = "force","momentum","Fixed")
# [x] PeriodicWall initialization ("x", "y", "z")
# [x] PlanarDomain initialization
# [x] NodeDomain initialization
# [x] NodeDomain initialization
# [ ] RigidParticles initialization ("positions" + "positions","frames","locations","rotations")

from numpy.testing import assert_allclose

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

    boundarycondition = PlanarDomain(axis0_friction=axis0_friction, axis1_friction=axis1_friction)

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

    boundarycondition = NodeDomain(axis0_mode=axis0_mode, axis1_mode=axis1_mode)

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
        gravity=gravity_start, is_ramp=True, ramp_step=10, gravity_end=gravity_end)

    assert isinstance(boundarycondition, Gravity)


def test_bodyforce_create():

    mask = np.array([False, True])
    if global_dimension == 1:
        values = np.array([[2.], [3.]])
    elif global_dimension == 2:
        values = np.array([[2., 7.], [3., 5.]])
    elif global_dimension == 3:
        values = np.array([[2., 7., 11], [3., 5., 10]])

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
        pos = np.array([[2.], [3.]])
        frames = np.array([0, 1])
        locations = np.array([[0.], [1.]])
        rotations = np.array([[0.], [0.]])
    elif global_dimension == 2:
        pos = np.array([[2., 7.], [3., 5.]])
        frames = np.array([0, 1])
        locations = np.array([[0., 2], [1., 3]])
        rotations = np.array([[0., 1], [0., 3]])
    elif global_dimension == 3:
        pos = np.array([[2., 7., 11], [3., 5., 10]])
        frames = np.array([0, 1])
        locations = np.array([[0., 2, 3], [1., 3, 4]])
        rotations = np.array([[0., 1, 0], [0., 3, 1]])
        
        
    boundarycondition = RigidParticles(pos)
    
    assert isinstance(boundarycondition, RigidParticles)
    
    boundarycondition = RigidParticles(pos,frames,locations,rotations)
    
    assert isinstance(boundarycondition, RigidParticles)
    