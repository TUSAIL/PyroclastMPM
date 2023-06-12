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
