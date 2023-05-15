

from pyroclastmpm import (
    NodesContainer,
    VTK,
    global_dimension
)

import numpy as np

# Functions tested
# [x] Initialize nodes
# [x] Get node coordinates

from numpy.testing import assert_allclose


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

    nodes = NodesContainer(node_start, node_end, node_spacing=0.2,output_formats=[VTK])
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
        expected_coords = np.array([[0., 0.],
                                    [0.2, 0.],
                                    [0.4, 0.],
                                    [0.6, 0.],
                                    [0.8, 0.],
                                    [1., 0.],
                                    [0., 0.2],
                                    [0.2, 0.2],
                                    [0.4, 0.2],
                                    [0.6, 0.2],
                                    [0.8, 0.2],
                                    [1., 0.2],
                                    [0., 0.4],
                                    [0.2, 0.4],
                                    [0.4, 0.4],
                                    [0.6, 0.4],
                                    [0.8, 0.4],
                                    [1., 0.4],
                                    [0., 0.6],
                                    [0.2, 0.6],
                                    [0.4, 0.6],
                                    [0.6, 0.6],
                                    [0.8, 0.6],
                                    [1., 0.6],
                                    [0., 0.8],
                                    [0.2, 0.8],
                                    [0.4, 0.8],
                                    [0.6, 0.8],
                                    [0.8, 0.8],
                                    [1., 0.8],
                                    [0., 1.],
                                    [0.2, 1.],
                                    [0.4, 1.],
                                    [0.6, 1.],
                                    [0.8, 1.],
                                    [1., 1.]])
        for ni, coord in enumerate(coords):
            assert_allclose(coord, expected_coords[ni])

    elif global_dimension == 3:
        expected_coords = np.array([[0., 0., 0.],
                                    [0.2, 0., 0.],
                                    [0.4, 0., 0.],
                                    [0., 0.2, 0.],
                                    [0.2, 0.2, 0.],
                                    [0.4, 0.2, 0.],
                                    [0., 0.4, 0.],
                                    [0.2, 0.4, 0.],
                                    [0.4, 0.4, 0.],
                                    [0., 0., 0.2],
                                    [0.2, 0., 0.2],
                                    [0.4, 0., 0.2],
                                    [0., 0.2, 0.2],
                                    [0.2, 0.2, 0.2],
                                    [0.4, 0.2, 0.2],
                                    [0., 0.4, 0.2],
                                    [0.2, 0.4, 0.2],
                                    [0.4, 0.4, 0.2],
                                    [0., 0., 0.4],
                                    [0.2, 0., 0.4],
                                    [0.4, 0., 0.4],
                                    [0., 0.2, 0.4],
                                    [0.2, 0.2, 0.4],
                                    [0.4, 0.2, 0.4],
                                    [0., 0.4, 0.4],
                                    [0.2, 0.4, 0.4],
                                    [0.4, 0.4, 0.4]])
        for ni, coord in enumerate(coords):
            assert_allclose(coord, expected_coords[ni])

