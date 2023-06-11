# Loading a config file and running a uniaxial stress test
import numpy as np
import tomllib
from pyroclastmpm import (
    CSV,
    MohrCoulomb,
    ParticlesContainer,
    global_dimension,
    set_global_timestep,
)

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

# check if code is compiled for correct dimension
if global_dimension != 3:
    raise ValueError(
        f"This example only works in {config['global']['dimension']}D. \
            The code is compiled for {global_dimension}D."
    )

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


print("Running uniaxial stress test")
"""
1. Uniaxial loading conditions
[ deps, 0, 0]
[ 0   , 0, 0]
[ 0   , 0, 0]

"""
particles, material = create_new_test()

deps = np.zeros((3, 3))

deps[0, 0] = config["uniaxial"]["deps_xx"]

particles.velocity_gradient = [deps]

stress_list, F_list, eps_e_list = [], [], []
for step in range(config["global"]["num_steps"]):
    # update deformation gradient
    particles.F = [
        (np.identity(3) + np.array(particles.velocity_gradient[0]) * dt)
        @ np.array(particles.F[0])
    ]
    particles, _ = material.stress_update(particles, 0)
    if step % config["global"]["output_steps"] == 0:
        stress_list.append(particles.stresses[0])
        F_list.append(particles.F[0])
        eps_e_list.append(material.eps_e[0])

stress_list = np.array(stress_list)
F_list = np.array(F_list)

np.save(config["uniaxial"]["output_directory"] + "stress.npy", stress_list)
np.save(config["uniaxial"]["output_directory"] + "F.npy", F_list)


print("Running simple stress test")
"""
2. Simple shear loading conditions, shear strain rate = deps_xy
[ 0, deps, 0]
[ deps, 0   , 0]
[ 0, 0   , 0]
"""
particles, material = create_new_test()

deps = np.zeros((3, 3))
deps[0, 1] = config["simpleshear"]["deps_xy"]
deps[1, 0] = config["simpleshear"]["deps_xy"]


particles.velocity_gradient = [deps]

stress_list, F_list, eps_e_list = [], [], []
for step in range(config["global"]["num_steps"]):
    # update deformation gradient
    particles.F = [
        (np.identity(3) + np.array(particles.velocity_gradient[0]) * dt)
        @ np.array(particles.F[0])
    ]
    particles, _ = material.stress_update(particles, 0)
    if step % config["global"]["output_steps"] == 0:
        stress_list.append(particles.stresses[0])
        F_list.append(particles.F[0])
        eps_e_list.append(material.eps_e[0])

stress_list = np.array(stress_list)
F_list = np.array(F_list)

np.save(config["simpleshear"]["output_directory"] + "stress.npy", stress_list)
np.save(config["simpleshear"]["output_directory"] + "F.npy", F_list)
