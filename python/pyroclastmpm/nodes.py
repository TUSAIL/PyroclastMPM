from __future__ import annotations

import numpy as np

from .pyroclastmpm_pybind import NodesContainer as PyroNodesContainer


class NodesContainer(PyroNodesContainer):
    """This container contains information on the background grid.

    Example usage:

    .. highlight:: python
    .. code-block:: python

        nodes = Nodes(
            node_start=np.array([0.0, 0.0, 0.0]),
            node_end=np.array([1.0, 1.0, 1.]),
            node_spacing= 1./40,
        )

    :param node_start: Coordinates for simulation start domain.
    :param node_end: Coordinates for simulation end domain.
    :param node_spacing: Grid spacing of background grid.

    """

    #: The start of the simulation domain (interfaced with pybind11)
    node_start: np.ndarray

    #: The end of the simulation domain (interfaced with pybind11) pybind11 binding for simulation end domain
    node_end: np.ndarray

    #: The background grid size (interfaced with pybind11)
    node_spacing: float

    #: The total number of nodes (interfaced with pybind11)
    num_nodes_total: int

    #: The nodal moments  (interfaced with pybind11)
    moments: np.ndarray

    #: The nodal moments incremented (USL) (interfaced with pybind11)
    moments_nt: np.ndarray

    #: External forces (e.g, gravity) (interfaced with pybind11)
    forces_external: np.ndarray

    #: Internal forces of the nodes  (interfaced with pybind11)
    forces_internal: np.ndarray

    #: Total forces (interfaced with pybind11)
    forces_total: np.ndarray

    #: Nodal masses (interfaced with pybind11)
    masses: np.ndarray

    def __init__(
        self,
        node_start: np.ndarray,
        node_end: np.ndarray,
        node_spacing: float,
        output_formats=[],
    ):
        """Initialize nodes container."""

        #: init c++ class from pybind11
        super(NodesContainer, self).__init__(
            node_start=node_start,
            node_end=node_end,
            node_spacing=node_spacing,
            output_formats=output_formats,
        )

    def give_coords(self) -> np.ndarray:
        """Give the nodal coordinates as a flat array

        :return: Nodal coordinates.
        """
        node_coords = super(NodesContainer, self).give_coords()
        return np.array(node_coords)
