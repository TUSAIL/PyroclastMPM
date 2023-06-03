from __future__ import annotations

import typing as t

import numpy as np

from . import pyroclastmpm_pybind as MPM


class NodesContainer(MPM.NodesContainer):
    """
    Node container containing the nodal coordinates.

    Example usage:

    .. highlight:: python
    .. code-block:: python

        nodes = Nodes(
            node_start=np.array([0.0, 0.0, 0.0]),
            node_end=np.array([1.0, 1.0, 1.]),
            node_spacing= 1./40,
        )

    """

    def __init__(
        self,
        node_start: np.ndarray,
        node_end: np.ndarray,
        node_spacing: float,
        output_formats: t.List[t.Type[MPM.OutputFormat]] = None,
    ):
        """Initialize Node Container

        Args:
            node_start (np.ndarray): Grid (domain) origin coordinate
            node_end (np.ndarray): Grid (domain) end coordinate
            node_spacing (float): spacing of the uniform grid
            output_formats (t.List[t.Type[MPM.OutputFormat]], optional): Output
            type. Defaults to None.
        """
        if output_formats is None:
            output_formats = []
        super(NodesContainer, self).__init__(
            node_start=node_start,
            node_end=node_end,
            node_spacing=node_spacing,
            output_formats=output_formats,
        )

    def give_coords(self) -> np.ndarray:
        """
        Returns the nodal coordinates

        Returns: Output the nodal coordinates
        """
        node_coords = super(NodesContainer, self).give_coords()
        return np.array(node_coords)
