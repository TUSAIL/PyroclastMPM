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


class BoundaryCondition(MPM.BoundaryCondition):
    """
    Base class of the boundary condition.
    Inherits from the C++ class through pybind11.
    """

    def __init__(self, *args, **kwargs):
        """Initialization of base class. Has no input parameters"""
        super(BoundaryCondition, self).__init__(*args, **kwargs)


class Gravity(MPM.Gravity):
    """
    Adds a gravitational force on the background nodes.
    The gravity is either ramped or constant.
    """

    def __init__(
        self,
        gravity: np.array,
        is_ramp: bool = False,
        ramp_step: int = 0,
        gravity_end: np.array = None,
    ):
        """
        Initialize gravitational boundary conditions
        applied on the Nodes.

        Args:
            gravity (np.array): Gravitational vector. 1D has shape (1,1),
                                2D has shape (1,2), 3D has shape (1,3)
            is_ramp (bool, optional): Flag to indicate if gravity ramps up.
                                      Defaults to False.
            ramp_step (int, optional): Number of steps until the end gravity
                                      is reached. Defaults to 0.
            gravity_end (np.array, optional): Gravity at the end of the ramp.
                                Same shape as 'gravity' arg. Defaults to None.
        """
        if gravity_end is None:
            gravity_end = np.zeros(MPM.global_dimension)
        else:
            gravity_end = np.array(gravity_end, ndmin=1)

        super(Gravity, self).__init__(
            gravity=gravity,
            is_ramp=is_ramp,
            ramp_step=ramp_step,
            gravity_end=gravity_end,
        )


class BodyForce(MPM.BodyForce):
    """
    Applies a body force boundary condition to background nodes
    """

    def __init__(self, mode: str, values: np.array, mask: np.array):
        """Initialize the body force boundary conditions

        Args:
            mode (str): 0 - Additive on force,
                        1 - Additive on momentum,
                        2 - Fixed on momentum.
            values (np.array): Values of the applied force.
                        should be shape (M,D), where M is the number of
                        masked nodes and D is the simulation dimension
            mask (np.array): Boolean mask of which nodes to apply the boundary
                        condition on
        """
        super(BodyForce, self).__init__(mode=mode, values=values, mask=mask)


class RigidBodyLevelSet(MPM.RigidBodyLevelSet):
    """
    Defines material and rigid body contact through a levelset
    """

    def __init__(
        self,
        COM: np.ndarray = None,
        frames: np.ndarray = None,
        locations: np.ndarray = None,
        rotations: np.ndarray = None,
        output_formats: t.List = None,
    ):
        """Initialize the RigidBodyLevelSet object.
        Note the rigid particles are defined in the ParticlesContainer.
        The animation file should be in .chan format.
        https://docs.blender.org/manual/en/latest/addons/import_export/anim_nuke_chan.html

        Args:
            COM (np.ndarray, optional): Center of mass of rigid body.
                            Only relevant to rigid body in motion.
                            Defaults to None.
            frames (np.ndarray, optional): Animation frames of rigid body.
                                    corresponds first parameter in .chan file.
                                    Defaults to None.
            locations (np.ndarray, optional): Locations of rigid body.
                                    corresponds to second parameter of .chan
                                    file. Defaults to None.
            rotations (np.ndarray, optional): Rotations of rigid body, its
                                    euler angles. Corresponds to third
                                    parameter of .chan file. Defaults to None.
            output_formats (t.List, optional):
                                    Output format of the stl (or particles).
                                    currently work in progress.
                                    Defaults to None.
        """
        if COM is None:
            COM = np.zeros(3)

        if frames is None:
            frames = []

        if locations is None:
            locations = []

        if rotations is None:
            rotations = []

        super(RigidBodyLevelSet, self).__init__(
            COM=COM,
            frames=frames,
            locations=locations,
            rotations=rotations,
            output_formats=output_formats,
        )


class PlanarDomain(MPM.PlanarDomain):
    """
    Domain of the MPM simulation with a boundary enforced
    by planar contact with friction
    """

    def __init__(
        self, axis0_friction: np.array = None, axis1_friction: np.array = None
    ):
        """Initialize a PlanarDomain boundary condition.

        Args:
            axis0_friction (np.array, optional):
                    Gives the friction of the (x0,y0,z0) plane  in the domain.
                    Has shape (1,D) where D is the dimension. Defaults to None.
            axis1_friction (np.array, optional):
                    Gives the friction of the (x1,y1,z1) plane  in the domain.
                    Has shape (1,D) where D is the dimension. Defaults to None.
        """
        if axis0_friction is None:
            axis0_friction = np.array = np.zeros(MPM.global_dimension)

        if axis1_friction is None:
            axis1_friction = np.zeros(MPM.global_dimension)

        super(PlanarDomain, self).__init__(
            axis0_friction=axis0_friction, axis1_friction=axis1_friction
        )


class NodeDomain(MPM.NodeDomain):
    """
    Domain of the MPM simulation with a boundary enforced
    through constrained on the node
    """

    def __init__(
        self, axis0_mode: np.array = None, axis1_mode: np.array = None
    ) -> None:
        """Initialize a NodeDomain boundary condition

        Args:
            axis0_mode (np.array, optional):
                    Gives the mode of the (x0,y0,z0) edge nodes in the domain.
                    Has shape (1,D) where D is the dimension.
                    0 - stick, 1-slip condition. Defaults to None.
            axis1_mode (np.array, optional):
                    Gives the mode of the (x1,y1,z1) edge nodes in the domain.
                    Has shape (1,D) where D is the dimension.
                    0 - stick, 1-slip condition. Defaults to None
        """
        if axis0_mode is None:
            axis0_mode = np.zeros(MPM.global_dimension, dtype=int)

        if axis1_mode is None:
            axis1_mode = (np.zeros(MPM.global_dimension, dtype=int),)
        super(NodeDomain, self).__init__(axis0_mode, axis1_mode)
