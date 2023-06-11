import pickle

import numpy as np
from numpy.testing import assert_allclose
from pyroclastmpm import VTK, ParticlesContainer, global_dimension

# Functions tested
# [x] Initialize particles
# [x] Pickle particles


def test_create_particles():
    if global_dimension == 1:
        pos = np.array([0])
    elif global_dimension == 2:
        pos = np.array([0, 1])
    elif global_dimension == 3:
        pos = np.array([0, 1, 2])

    particles = ParticlesContainer(positions=[pos])
    assert isinstance(particles, ParticlesContainer)


def test_pickle_particles():
    """Test that a material can be pickled."""

    if global_dimension == 1:
        pos = np.array([0])
    elif global_dimension == 2:
        pos = np.array([0, 1])
    elif global_dimension == 3:
        pos = np.array([0, 1, 2])

    particles = ParticlesContainer(positions=[pos], output_formats=[VTK])

    # check dump and load material using pickle
    filename = "particles.pkl"
    with open(filename, "wb") as f:
        pickle.dump(particles, f)

    with open(filename, "rb") as f:
        particles = pickle.load(f)

    assert_allclose(particles.positions[0], pos)
