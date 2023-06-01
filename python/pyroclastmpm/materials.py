from __future__ import annotations

from .pyroclastmpm_pybind import LinearElastic as PyroLinearElastic
from .pyroclastmpm_pybind import LocalGranularRheology as PyroLocalGranularRheology
from .pyroclastmpm_pybind import Material as PyroMaterial
from .pyroclastmpm_pybind import NewtonFluid as PyroNewtonFluid
from .pyroclastmpm_pybind import VonMises as PyroVonMises


class Material(PyroMaterial):
    """Base class of the material. Inherits from the C++ class through pybind11."""

    def __init__(self, *args, **kwargs):
        """Initialization of base class. Has no input parameters."""
        super(Material, self).__init__(*args, **kwargs)


class LinearElastic(PyroLinearElastic):
    """Linear Elastic Material inheritrs from the C++ class through pybind11."""

    #: pybind11 binding for material name
    name: str

    #: pybind11 binding for Young's modulus
    density: float

    #: pybind11 binding for Young's modulus
    E: float

    #: pybind11 binding for Poisson's ratio
    pois: float

    #: pybind11 binding for shear modulus
    shear_modulus: float

    #: pybind11 binding for lame modulus
    lame_modulus: float

    def __init__(self, density: float, E: float, pois: float = 0):  # NOSONAR
        """_summary_

        :param density: Material density
        :param E: Youngs modulus
        :param pois: Poisson's ratio, defaults to 0
        """
        super(LinearElastic, self).__init__(density=density, E=E, pois=pois)


class VonMises(PyroVonMises):
    """Associated Drucker Prager inheritrs from the C++ class through pybind11."""

    #: pybind11 binding for material name
    name: str

    #: pybind11 binding for Young's modulus
    density: float

    #: pybind11 binding for Young's modulus
    E: float

    #: pybind11 binding for Poisson's ratio
    pois: float

    #: pybind11 binding for shear modulus
    shear_modulus: float

    #: pybind11 binding for lame modulus
    lame_modulus: float

    def __init__(
        self,
        density: float,
        E: float,
        pois: float = 0,
        yield_stress: float = 0,
        H: float = 1,
    ):  # NOSONAR
        """_summary_

        :param density: Material density
        :param E: Youngs modulus
        :param pois: Poisson's ratio, defaults to 0
        """
        super(VonMises, self).__init__(
            density=density, E=E, pois=pois, yield_stress=yield_stress, H=H
        )


class NewtonFluid(PyroNewtonFluid):
    #: pybind11 binding for material name
    name: str

    #: pybind binding for density of the fluid
    density: float

    #: pybind binding for viscosity of the fluid
    viscocity: float

    #: pybind binding for bulk modulus of the fluid
    bulk_modulus: float

    #: pybind binding for gamma parameter of the fluid
    gamma: float

    def __init__(
        self,
        density: float,
        viscosity: float,
        bulk_modulus: float = 0.0,
        gamma: float = 7.0,
    ):
        super(NewtonFluid, self).__init__(
            density=density, viscosity=viscosity, bulk_modulus=bulk_modulus, gamma=gamma
        )


class LocalGranularRheology(PyroLocalGranularRheology):
    """Local Granular Rheology inheritrs from the C++ class through pybind11."""

    #: pybind11 binding for material name
    name: str

    #: pybind binding for density of the granular material
    density: float

    #: pybind11 binding for Young's modulus
    E: float

    #: pybind11 binding for Poisson's ratio
    pois: float

    #: pybind11 binding for shear modulus
    shear_modulus: float

    #: pybind11 binding for lame modulus
    lame_modulus: float

    #: pybind11 binding for inertial number
    I0: float

    #: pybind11 binding for static friction coefficient
    mu_s: float

    #: pybind11 binding for dynamic friction coefficient
    mu_2: float

    #: pybind11 binding for critical density
    rho_c: float

    #: pybind11 binding for particle diameter
    particle_diameter: float

    #: pybind11 binding for particle density
    particle_density: float

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
    ):  #  NOSONAR
        """_summary_

        :param E: Youngs modulus
        :param pois: Poisson's ratio, defaults to 0
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
