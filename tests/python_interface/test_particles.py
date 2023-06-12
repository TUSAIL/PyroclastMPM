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
