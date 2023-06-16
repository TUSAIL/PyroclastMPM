# Loading a config file and running a uniaxial stress test
import tomllib

import numpy as np
from constitutive_analysis import servo_control
from pyroclastmpm import (
    CSV,
    MohrCoulomb,
    ParticlesContainer,
    check_dimension,
    set_global_timestep,
)

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

check_dimension(3)

# Time step for increment of deformation gradient

dt = config["global"]["timestep"]
set_global_timestep(dt)


# we use a single material point to simulate uniaxial stress
def create_new_test():
    particles = ParticlesContainer(
        positions=np.array([[0.0, 0.0, 0.0]]), output_formats=[CSV]
    )
    # initialize material
    material = MohrCoulomb(
        config["material"]["density"],
        config["material"]["E"],
        config["material"]["pois"],
        config["material"]["cohesion"],
        config["material"]["friction_angle"],
        config["material"]["dilatancy_angle"],
        config["material"]["H"],
    )
    particles, _ = material.initialize(particles, 0)
    return particles, material


stress_list, F_list, eps_e_list = np.array([]), np.array([]), np.array([])


def store_stress(particles, material, step):
    global stress_list, F_list, eps_e_list

    F = np.array(particles.F[0])
    strain = 0.5 * (F.T + F) - np.identity(3)
    print("stress", particles.stresses[0])

    print("strain", strain)


particles, material = create_new_test()


particles, material = servo_control(
    particles,
    material,
    config["uniaxial"]["deps_xx"] * dt,
    ["stress", "stress", "stress"],
    [-1000, -1000, -1000],
    1e-3,
    10,
    store_stress,
)


particles, material = servo_control(
    particles,
    material,
    config["uniaxial"]["deps_xx"] * dt,
    ["stress", "strain", "stress"],
    [-1000, -0.2, -1000],
    1e-3,
    10,
    store_stress,
)
