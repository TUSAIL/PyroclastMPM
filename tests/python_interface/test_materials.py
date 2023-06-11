import pickle

from pyroclastmpm import (
    DruckerPrager,
    LinearElastic,
    LocalGranularRheology,
    Material,
    NewtonFluid,
)

# Features to test
# [x] material initialize
# [x] material pickle
# [x] linearelastic initialize
# [x] linearelastic pickle
# [x] newtonfluid initialize
# [x] newtonfluid pickle
# [x] localrheo initialize
# [x] localrheo pickle
# [x] druckerprager initialize
# [x] druckerprager pickle


def test_pickle_material():
    """Test that a material can be pickled."""
    material = Material()

    # check pybind11 set default values
    assert material.name == "None"
    assert material.density == 0.0

    # check if we can set values
    material.name = "Foo"
    material.density = 1.0

    # check dump and load material using pickle
    filename = "material.pkl"
    with open(filename, "wb") as f:
        pickle.dump(material, f)

    with open(filename, "rb") as f:
        material = pickle.load(f)

    # check if values are still set
    assert material.name == "Foo"
    assert material.density == 1.0


def test_pickle_linearelastic():
    """Test that a LinearElastic material can be pickled."""
    material = LinearElastic(density=1.0, E=1.0, pois=0.0)

    # check pybind11 set default values
    assert material.name == "LinearElastic"
    assert material.density == 1.0
    assert material.E == 1.0
    assert material.pois == 0.0

    # check if we can set values
    material.name = "Foo"
    material.density = 2.0
    material.E = 2.0
    material.pois = 1.0

    lam = material.lame_modulus
    G = material.shear_modulus
    K = material.bulk_modulus

    # check dump and load material using pickle
    filename = "linearelastic.pkl"
    with open(filename, "wb") as f:
        pickle.dump(material, f)

    with open(filename, "rb") as f:
        material = pickle.load(f)

    # check if values are still set
    assert material.name == "Foo"
    assert material.density == 2.0
    assert material.E == 2.0
    assert material.pois == 1.0

    # check if G,K,lambda are set correctly
    assert material.lame_modulus == lam
    assert material.bulk_modulus == K
    assert material.shear_modulus == G


def test_pickle_newtonfluid():
    material = NewtonFluid(
        density=1.0, viscosity=1.0, bulk_modulus=1.0, gamma=1.0
    )

    # check pybind11 set default values
    assert material.name == "NewtonFluid"
    assert material.density == 1.0
    assert material.viscosity == 1.0
    assert material.bulk_modulus == 1.0
    assert material.gamma == 1.0

    # check if we can set values
    material.name = "Foo"
    material.density = 2.0
    material.viscosity = 2.0
    material.bulk_modulus = 2.0
    material.gamma = 2.0

    # check dump and load material using pickle
    filename = "newtonfluid.pkl"
    with open(filename, "wb") as f:
        pickle.dump(material, f)

    with open(filename, "rb") as f:
        material = pickle.load(f)

    # check if values are still set
    assert material.name == "Foo"
    assert material.density == 2.0
    assert material.viscosity == 2.0
    assert material.bulk_modulus == 2.0
    assert material.gamma == 2.0


def test_pickle_localrheology():
    material = LocalGranularRheology(
        density=1.0,
        E=1.0,
        pois=2.0,
        I0=3.0,
        mu_s=3.0,
        mu_2=3.0,
        rho_c=3.0,
        particle_diameter=2.0,
        particle_density=2.0,
    )

    # check pybind11 set default values
    assert material.name == "LocalGranularRheology"
    assert material.density == 1.0
    assert material.E == 1.0
    assert material.pois == 2.0
    assert material.I0 == 3.0
    assert material.mu_s == 3.0
    assert material.mu_2 == 3.0
    assert material.rho_c == 3.0
    assert material.particle_diameter == 2.0
    assert material.particle_density == 2.0

    # check if we can set values
    material.name = "Foo"
    material.density = 2.0
    material.E = 2.0
    material.pois = 1.0
    material.I0 = 2.0
    material.mu_s = 2.0
    material.mu_2 = 2.0
    material.rho_c = 2.0
    material.particle_diameter = 1.0
    material.particle_density = 1.0

    # check dump and load material using pickle
    filename = "localgranularrheology.pkl"
    with open(filename, "wb") as f:
        pickle.dump(material, f)

    with open(filename, "rb") as f:
        material = pickle.load(f)

    # check if values are still set
    assert material.name == "Foo"
    assert material.density == 2.0
    assert material.E == 2.0
    assert material.pois == 1.0
    assert material.I0 == 2.0
    assert material.mu_s == 2.0
    assert material.mu_2 == 2.0
    assert material.rho_c == 2.0
    assert material.particle_diameter == 1.0
    assert material.particle_density == 1.0

    lam = material.lame_modulus
    G = material.shear_modulus
    K = material.bulk_modulus

    # check if G,K,lambda are set correctly
    assert material.lame_modulus == lam
    assert material.bulk_modulus == K
    assert material.shear_modulus == G


def test_pickle_druckerprager():
    material = DruckerPrager(1, 0.2, 1, 3, 4, 5)

    # check if we can set values
    material.name = "Foo"
    material.density = 2.0
    material.E = 2.0
    material.pois = 1.0
    material.friction_angle = 2.0
    material.vcs = 1.0
    material.cohesion = 2.0
    # check dump and load material using pickle
    filename = "druckerprager.pkl"
    with open(filename, "wb") as f:
        pickle.dump(material, f)

    with open(filename, "rb") as f:
        material = pickle.load(f)

    # check if values are still set
    assert material.name == "Foo"
    assert material.density == 2.0
    assert material.E == 2.0
    assert material.pois == 1.0
    assert material.friction_angle == 2.0
    assert material.vcs == 1.0
    assert material.cohesion == 2.0
