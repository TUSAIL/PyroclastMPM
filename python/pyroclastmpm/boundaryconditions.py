from __future__ import annotations

import numpy as np

import typing as t

from .pyroclastmpm_pybind import global_dimension

from .pyroclastmpm_pybind import (
    BoundaryCondition as PyroBoundaryCondition,
)
from .pyroclastmpm_pybind import (
    Gravity as PyroGravity,
)

from .pyroclastmpm_pybind import (
    BodyForce as PyroBodyForce,
)

from .pyroclastmpm_pybind import (
    RigidParticles as PyroRigidParticles,
)

from .pyroclastmpm_pybind import (
    PlanarDomain as PyroPlanarDomain,
)

from .pyroclastmpm_pybind import (
    NodeDomain as PyroNodeDomain,
)


class BoundaryCondition(PyroBoundaryCondition):
    """Base class of the boundary condition. Inherits from the C++ class through pybind11."""

    def __init__(self, *args, **kwargs):
        """Initialization of base class. Has no input parameters"""
        super(BoundaryCondition, self).__init__(*args, **kwargs)


class Gravity(PyroGravity):

    gravity: np.array

    def __init__(
        self,
        gravity: np.array,
        is_ramp: bool = False,
        ramp_step: int = 0,
        gravity_end: np.array = np.zeros(global_dimension),
    ):
        _gravity_end = np.array(gravity_end, ndmin=1)

        super(Gravity, self).__init__(
            gravity=gravity,
            is_ramp=is_ramp,
            ramp_step=ramp_step,
            gravity_end=_gravity_end,
        )


class BodyForce(PyroBodyForce):

    #: mode of boundary condition applied (0 - additve on forces, 1 - additive on momentum, 2 - fixed on momentum)
    mode_id: int

    def __init__(self, mode: str, values: np.array, mask: np.array):
        super(BodyForce, self).__init__(mode=mode, values=values, mask=mask)


class RigidParticles(PyroRigidParticles):
    def __init__(
        self,
        positions: np.ndarray,
        frames: np.ndarray = [],
        locations: np.ndarray = [],
        rotations: np.ndarray = [],
        output_formats=[],
    ):
        super(RigidParticles, self).__init__(
            positions=positions, frames=frames, locations=locations, rotations=rotations, output_formats=output_formats,
        )


class PlanarDomain(PyroPlanarDomain):

    def __init__(
            self,
            axis0_friction: np.array = np.zeros(global_dimension),
            axis1_friction: np.array = np.zeros(global_dimension)):
        super(PlanarDomain, self).__init__(
            axis0_friction=axis0_friction,
            axis1_friction=axis1_friction
        )


class NodeDomain(PyroNodeDomain):

    def __init__(
            self,
            axis0_mode: np.array = np.zeros(global_dimension, dtype=int),
            axis1_mode: np.array = np.zeros(global_dimension, dtype=int)):
        super(NodeDomain, self).__init__(
            axis0_mode=axis0_mode,
            axis1_mode=axis1_mode
        )
