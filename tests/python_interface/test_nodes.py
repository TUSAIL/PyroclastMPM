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
from numpy.testing import assert_allclose
from pyroclastmpm import VTK, NodesContainer, global_dimension

# Functions tested
# [x] Initialize nodes
# [x] Get node coordinates


def test_nodes_particles():
    if global_dimension == 1:
        node_start = np.array([0])
        node_end = np.array([1])
    elif global_dimension == 2:
        node_start = np.array([0, 0])
        node_end = np.array([1, 1])
    elif global_dimension == 3:
        node_start = np.array([0, 0, 0])
        # 0.4 is to avoid too many nodes (slow testing + long code)
        node_end = np.array([0.4, 0.4, 0.4])

    nodes = NodesContainer(
        node_start, node_end, node_spacing=0.2, output_formats=[VTK]
    )
    assert isinstance(nodes, NodesContainer)

    coords = nodes.give_coords()

    if global_dimension == 1:
        assert_allclose(coords[0], 0)
        assert_allclose(coords[1], 0.2)
        assert_allclose(coords[2], 0.4)
        assert_allclose(coords[3], 0.6)
        assert_allclose(coords[4], 0.8)
        assert_allclose(coords[5], 1)

    elif global_dimension == 2:
        expected_coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.4, 0.0],
                [0.6, 0.0],
                [0.8, 0.0],
                [1.0, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [0.4, 0.2],
                [0.6, 0.2],
                [0.8, 0.2],
                [1.0, 0.2],
                [0.0, 0.4],
                [0.2, 0.4],
                [0.4, 0.4],
                [0.6, 0.4],
                [0.8, 0.4],
                [1.0, 0.4],
                [0.0, 0.6],
                [0.2, 0.6],
                [0.4, 0.6],
                [0.6, 0.6],
                [0.8, 0.6],
                [1.0, 0.6],
                [0.0, 0.8],
                [0.2, 0.8],
                [0.4, 0.8],
                [0.6, 0.8],
                [0.8, 0.8],
                [1.0, 0.8],
                [0.0, 1.0],
                [0.2, 1.0],
                [0.4, 1.0],
                [0.6, 1.0],
                [0.8, 1.0],
                [1.0, 1.0],
            ]
        )
        for ni, coord in enumerate(coords):
            assert_allclose(coord, expected_coords[ni])

    elif global_dimension == 3:
        expected_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.2, 0.2, 0.0],
                [0.4, 0.2, 0.0],
                [0.0, 0.4, 0.0],
                [0.2, 0.4, 0.0],
                [0.4, 0.4, 0.0],
                [0.0, 0.0, 0.2],
                [0.2, 0.0, 0.2],
                [0.4, 0.0, 0.2],
                [0.0, 0.2, 0.2],
                [0.2, 0.2, 0.2],
                [0.4, 0.2, 0.2],
                [0.0, 0.4, 0.2],
                [0.2, 0.4, 0.2],
                [0.4, 0.4, 0.2],
                [0.0, 0.0, 0.4],
                [0.2, 0.0, 0.4],
                [0.4, 0.0, 0.4],
                [0.0, 0.2, 0.4],
                [0.2, 0.2, 0.4],
                [0.4, 0.2, 0.4],
                [0.0, 0.4, 0.4],
                [0.2, 0.4, 0.4],
                [0.4, 0.4, 0.4],
            ]
        )
        for ni, coord in enumerate(coords):
            assert_allclose(coord, expected_coords[ni])
