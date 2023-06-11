# Loading a config file and running a simple shear stress test
import tomllib

import numpy as np
from pyroclastmpm import (
    CSV,
    LinearElastic,
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
        f"""
        This example only works in {config['global']['dimension']}D.
          The code is compiled for {global_dimension}D."""
    )

# Time step for increment of deformation gradient
dt = config["global"]["timestep"]
set_global_timestep(dt)

# we use a single material point to simulate uniaxial stress
particles = ParticlesContainer(
    positions=np.array([[0.0, 0.0, 0.0]]), output_formats=[CSV]
)

material = LinearElastic(
    config["material"]["density"],
    config["material"]["E"],
    config["material"]["pois"],
)

# Simple shear loading conditions, shear strain rate = deps_xy
# [ 0, deps, 0]
# [ 0, 0   , 0]
# [ 0, 0   , 0]
deps = np.zeros((3, 3))
deps[0, 1] = config["simpleshear"]["deps_xy"]
particles.velocity_gradient = [deps]

# initial deformation gradient
F = np.identity(3)

stress_list, F_list = [], []
for step in range(config["global"]["num_steps"]):
    particles, _ = material.stress_update(particles, 0)
    F = (np.identity(3) + np.array(particles.velocity_gradient[0]) * dt) @ F
    if step % config["global"]["output_steps"] == 0:
        stress_list.append(particles.stresses[0])
        F_list.append(F)

stress_list = np.array(stress_list)
F_list = np.array(F_list)

np.save(config["simpleshear"]["output_directory"] + "stress.npy", stress_list)
np.save(config["simpleshear"]["output_directory"] + "F.npy", F_list)
