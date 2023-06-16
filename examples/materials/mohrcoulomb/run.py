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
    # print(strain)
    # stress_list = np.append(stress_list, particles.stresses[0])
    # F_list = np.append(F_list, particles.F[0])
    # eps_e_list = np.append(eps_e_list, material.eps_e[0])


particles, material = create_new_test()

# def servo_control(
#     particles,
#     material,
#     max_load_rate,
#     control,
#     targets,
#     callback=None,
#     accuracy=1e-3,
#     dim=3,
# ):

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

# isotropic_compression(
#     particles,
#     material,
#     config["uniaxial"]["deps_xx"] * dt,
#     config["global"]["total_steps"],
#     store_stress,
# )

# print("Stress list:", stress_list)

# stress_list = np.array(stress_list)
# F_list = np.array(F_list)

# np.save(config["uniaxial"]["output_directory"] + "stress.npy", stress_list)
# np.save(config["uniaxial"]["output_directory"] + "F.npy", F_list)


# print("Running simple stress test")
# """
# 2. Simple shear loading conditions, shear strain rate = deps_xy
# [ 0, deps, 0]
# [ deps, 0   , 0]
# [ 0, 0   , 0]
# """
# particles, material = create_new_test()

# deps = np.zeros((3, 3))
# deps[0, 1] = config["simpleshear"]["deps_xy"]
# deps[1, 0] = config["simpleshear"]["deps_xy"]


# particles.velocity_gradient = [deps]

# stress_list, F_list, eps_e_list = [], [], []
# for step in range(config["global"]["num_steps"]):
#     # update deformation gradient
#     particles.F = [
#         (np.identity(3) + np.array(particles.velocity_gradient[0]) * dt)
#         @ np.array(particles.F[0])
#     ]
#     particles, _ = material.stress_update(particles, 0)
#     if step % config["global"]["output_steps"] == 0:
#         stress_list.append(particles.stresses[0])
#         F_list.append(particles.F[0])
#         eps_e_list.append(material.eps_e[0])

# stress_list = np.array(stress_list)
# F_list = np.array(F_list)

# np.save(config["simpleshear"]["output_directory"] + "stress.npy", stress_list)
# np.save(config["simpleshear"]["output_directory"] + "F.npy", F_list)
