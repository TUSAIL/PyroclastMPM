
import pytest

from pyroclastmpm import (
    ParticlesContainer,
    NodesContainer,
    USL,
    TLMPM,
    MUSL,
    Material,
    BoundaryCondition,
    global_dimension
)

import numpy as np

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
    solver = solverclass(particles, nodes, materials=[
        dummy_material], boundaryconditions=[dummy_boundarycondition], alpha=0.5)
    assert isinstance(solver, solverclass)
