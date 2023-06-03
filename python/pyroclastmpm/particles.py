from __future__ import annotations

import typing as t

import numpy as np

from . import pyroclastmpm_pybind as MPM


class ParticlesContainer(MPM.ParticlesContainer):
    """Particles container class"""

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray = None,
        colors: t.List[int] = None,
        is_rigid: t.List[bool] = None,
        stresses: np.ndarray = None,
        masses: np.ndarray = None,
        volumes: np.ndarray = None,
        output_formats: t.List[t.Type[MPM.OutputFormat]] = None,
    ):
        """Initialize Particles Container

        Args:
            positions (np.ndarray): Coordinates of the particles of
                                    shape (N, D) where N is the number
                                    of particles and D is the dimension
                                    of the problem
            velocities (np.ndarray, optional): Velocities of the particles
                                    of the same shape as positions.
                                    Defaults to None.
            colors (t.List[int], optional): Colors or material type of the
                                    particle of shape (N). Defaults to None.
            is_rigid (t.List[bool], optional): Mask if particles are rigid
                                    or not. Defaults to None.
            stresses (np.ndarray, optional): Initial stress of particles of
                                    shape (N,D,D). Defaults to None.
            masses (np.ndarray, optional): Initial mass of particles of
                                    shape (N). Defaults to None.
            volumes (np.ndarray, optional): Initial volume of particles
                                    of shape (N). Defaults to None.
            output_formats (t.List[t.Type[MPM.OutputFormat]], optional):
                                    Output type (e.g VTK, CSV, OBJ).
                                    Defaults to None.
        """
        if output_formats is None:
            output_formats = []
        if colors is None:
            colors = []
        if is_rigid is None:
            is_rigid = []
        if velocities is None:
            velocities = []
        if stresses is None:
            stresses = []
        if masses is None:
            masses = []
        if volumes is None:
            volumes = []

        super(ParticlesContainer, self).__init__(
            positions=positions,
            velocities=velocities,
            colors=colors,
            is_rigid=is_rigid,
            stresses=stresses,
            masses=masses,
            volumes=volumes,
            output_formats=output_formats,
        )
