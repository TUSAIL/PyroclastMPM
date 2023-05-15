from __future__ import annotations

from .pyroclastmpm_pybind import ParticlesContainer as PyroParticlesContainer

import numpy as np
from typing import Type, List


class ParticlesContainer(PyroParticlesContainer):
    """ParticleContainer wrapper for C++ class."""

    #: pybind11 binding for number of particles
    num_particles: int

    #: pybind11 binding for particle positions
    positions: np.ndarray

    #: pybind11 binding for particle velocities
    velocities: np.ndarray

    #: pybind11 binding for particle stresses
    stresses: np.ndarray

    #: pybind11 binding for particle strains
    strains: np.ndarray

    #: pybind11 binding for particle strain rates
    strain_rates: np.ndarray

    #: pybind11 binding for particle deformation matricies
    F: np.ndarray

    #: pybind11 binding for particle velocity gradients
    velocity_gradient: np.ndarray

    #: pybind11 binding for particle pressures
    pressures: np.ndarray

    #: pybind11 binding for particle masses
    masses: np.ndarray

    #: pybind11 binding for particle volumes
    volumes: np.ndarray

    #: pybind11 binding for particle original volumes
    volumes_original: np.ndarray

    #: pybind11 binding for particle colors (or material types)
    colors: np.ndarray

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray = [],
        colors: List[int] = [],
        stresses: np.ndarray = [],
        masses: np.ndarray = [],
        volumes: np.ndarray = [],
        output_formats=[],
    ):
        """Initialize particles container class

        :param positions: particle positions
        :param masses: particle masses
        :params colors: list of particle colors (or material types), depending on materials, defaults to []
        :param stresses: particle stresses
        :param masses: particle masses
        :param volumes: particle volumes
        """
        #: init c++ class from pybind11
        super(ParticlesContainer, self).__init__(
            positions=positions,
            velocities=velocities,
            colors=colors,
            stresses=stresses,
            masses=masses,
            volumes=volumes,
            output_formats=output_formats,
        )
