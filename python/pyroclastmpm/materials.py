from __future__ import annotations

import typing as t

from . import pyroclastmpm_pybind as MPM


class Material(MPM.Material):
    """Base class of the material"""

    def __init__(self, *args, **kwargs):
        """Initialization of base class. Has no input parameters."""
        super(Material, self).__init__(*args, **kwargs)


class LinearElastic(MPM.LinearElastic):
    """Isotropic Linear Elasticity"""

    def __init__(self, density: float, E: float, pois: float = 0):
        """Initialize isotropic linear elastic material

        Args:
            density (float): Bulk density
            E (float): Young's modulus
            pois (float, optional): Poisson's ratio. Defaults to 0.
        """
        super(LinearElastic, self).__init__(density=density, E=E, pois=pois)


class VonMises(MPM.VonMises):
    """Associative Von Mises plasticity (linear isotropic strain hardening)


    de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen.
    Computational methods for plasticity: theory and applications.
    John Wiley & Sons, 2011.
    """

    def __init__(
        self,
        density: float,
        E: float,
        pois: float = 0,
        yield_stress: float = 0,
        H: float = 1,
    ):
        """Initialize Associative Von Mises plasticity

        Args:
            density (float): Bulk density
            E (float): Young's modulus
            pois (float, optional): Poisson's ratio. Defaults to 0.
            yield_stress (float, optional): Initial yield stress.
                                            Defaults to 0.
            H (float, optional): Hardening modulus. Defaults to 1.
        """
        super(VonMises, self).__init__(
            density=density, E=E, pois=pois, yield_stress=yield_stress, H=H
        )

    def initialize(
        self, particles: t.Type[MPM.ParticlesContainer], mat_id: int = 0
    ) -> t.Type[MPM.ParticlesContainer]:
        """Initialize internal variables
        (need to allocate memory for internal purposes)

        Args:
            particles (t.Type[MPM.ParticlesContainer]): Particle container
            mat_id (int, optional): Material ID. Defaults to 0.

        Returns:
            t.Type[MPM.ParticlesContainer]: Particle container (initialized)
        """
        return super(VonMises, self).initialize(particles, mat_id)

    def stress_update(
        self, particles: t.Type[MPM.ParticlesContainer], mat_id: int = 0
    ) -> t.Type[MPM.ParticlesContainer]:
        """

        Perform a stress update step

        Args:
            particles (t.Type[MPM.ParticlesContainer]): Particle container
            mat_id (int, optional): Material ID. Defaults to 0.

        Returns:
            t.Type[MPM.ParticlesContainer]: Particle container (initialized)
            int: Material ID
        """
        return super(VonMises, self).stress_update(particles, mat_id)


class MohrCoulomb(MPM.MohrCoulomb):
    """
    Non-Associative Mohr-Coulomb plasticity with
    isotropic strain hardening

    de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen.
    Computational methods for plasticity: theory and applications.
    John Wiley & Sons, 2011.
    """

    def __init__(
        self,
        density: float,
        E: float,
        pois: float = 0,
        cohesion: float = 0,
        friction_angle: float = 0,
        dilatancy_angle: float = 0,
        H: float = 1,
    ):
        """Initialize Non-Associative Mohr-Coulomb plasticity

        Args:
            density (float): Bulk density
            E (float): Young's modulus
            pois (float, optional): Poisson's ratio.
                                    Defaults to 0.
            cohesion (float, optional): Cohesion.
                                        Defaults to 0.
            friction_angle (float, optional): friction angle (in degrees).
                                              Defaults to 0.
            dilatancy_angle (float, optional): dilatancy angle (in degrees).
                                              Defaults to 0.
            H (float, optional): Hardening modulus.
                                Defaults to 1.
        """

        super(MohrCoulomb, self).__init__(
            density=density,
            E=E,
            pois=pois,
            cohesion=cohesion,
            friction_angle=friction_angle,
            dilatancy_angle=dilatancy_angle,
            H=H,
        )

    def initialize(
        self, particles: t.Type[MPM.ParticlesContainer], mat_id: int = 0
    ) -> t.Type[MPM.ParticlesContainer]:
        """Initialize internal variables
        (need to allocate memory for internal purposes)

        Args:
            particles (t.Type[MPM.ParticlesContainer]): Particle container
            mat_id (int, optional): Material ID. Defaults to 0.

        Returns:
            t.Type[MPM.ParticlesContainer]: Particle container (initialized)
        """
        return super(MohrCoulomb, self).initialize(particles, mat_id)

    def stress_update(
        self, particles: t.Type[MPM.ParticlesContainer], mat_id: int = 0
    ) -> t.Type[MPM.ParticlesContainer]:
        """

        Perform a stress update step

        Args:
            particles (t.Type[MPM.ParticlesContainer]): Particle container
            mat_id (int, optional): Material ID. Defaults to 0.

        Returns:
            t.Type[MPM.ParticlesContainer]: Particle container (initialized)
            int: Material ID
        """
        return super(MohrCoulomb, self).stress_update(particles, mat_id)


class NewtonFluid(MPM.NewtonFluid):
    """Newtonian fluid"""

    def __init__(
        self,
        density: float,
        viscosity: float,
        bulk_modulus: float = 0.0,
        gamma: float = 7.0,
    ):
        """Newtonian fluid

        Args:
            density (float): Bulk density
            viscosity (float): viscocity of the fluid (mu)
            bulk_modulus (float, optional): Bulk modulus of the fluid.
                                            Defaults to 0.0.
            gamma (float, optional): gamma parameter related to the
                                     equation of state (EOS).
                                     Defaults to 7.0.
        """
        super(NewtonFluid, self).__init__(
            density=density,
            viscosity=viscosity,
            bulk_modulus=bulk_modulus,
            gamma=gamma,
        )


class LocalGranularRheology(MPM.LocalGranularRheology):
    """Local rheological Hypoelasticity (warning unstable)


    Dunatunga, Sachith, and Ken Kamrin.
    Continuum modelling and simulation of granular flows
    through their many phases.
    Journal of Fluid Mechanics 779 (2015): 483-513.
    """

    def __init__(
        self,
        density: float,
        E: float,
        pois: float,
        I0: float,
        mu_s: float,
        mu_2: float,
        rho_c: float,
        particle_diameter: float,
        particle_density: float,
    ):
        """Initialize hypoelastic local rheology (warning unstable)

        Args:
            density (float): Bulk density
            E (float): Young's modulus
            pois (float): Poisson's ratio
            I0 (float): Fitting parameter
            mu_s (float): Fitting parameter
            mu_2 (float): Fitting parameter
            rho_c (float): Critical density
            particle_diameter (float): Particle diameter
            particle_density (float): Particle density
        """
        super(LocalGranularRheology, self).__init__(
            density=density,
            E=E,
            pois=pois,
            I0=I0,
            mu_s=mu_s,
            mu_2=mu_2,
            rho_c=rho_c,
            particle_diameter=particle_diameter,
            particle_density=particle_density,
        )
