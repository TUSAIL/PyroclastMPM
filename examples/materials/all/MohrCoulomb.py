# Loading a config file and running a uniaxial stress test
import tomllib

import numpy as np
import plot
import xarray as xr
from constitutive_analysis.loading import strain_controlled
from pyroclastmpm.MPM3D import MohrCoulomb, ParticlesContainer, set_global_timestep

# load config file
with open("./config.toml", "rb") as f:
    cfg = tomllib.load(f)

# Time step for increment of deformation gradient
dt = cfg["global"]["timestep"]
set_global_timestep(dt)

# Create particles and initialize material
particle = ParticlesContainer([[0.0, 0.0, 0.0]])

material = MohrCoulomb(
    cfg["mohr_coulomb"]["density"],
    cfg["mohr_coulomb"]["E"],
    cfg["mohr_coulomb"]["pois"],
    cfg["mohr_coulomb"]["cohesion"],
    cfg["mohr_coulomb"]["friction_angle"],
    cfg["mohr_coulomb"]["dilatancy_angle"],
    cfg["mohr_coulomb"]["H"],
)

particle, _ = material.initialize(particle, 0)

deps = np.zeros((3, 3))

deps[0, 0] = np.array(cfg["uniaxial"]["deps_xx"])


data = {"stress": [], "strain": []}


def callback_getarrs(particles, _, step):
    global data
    if step % cfg["global"]["output_steps"] == 0:
        data["stress"].append(particles.stresses[0])
        F = np.array(particles.F[0])
        strain = 0.5 * (F.T + F) - np.identity(3)
        data["strain"].append(strain)


list_datasets = []

# list_datasets.append(ds)

# run uniaxial stress test
particles, material = strain_controlled(
    particle,
    material,
    deps,
    cfg["global"]["num_steps"],
    dt,
    callback_getarrs,
)

ds = xr.Dataset(
    data_vars=dict(
        stress=(["step", "row", "col"], data["stress"]),
        strain=(["step", "row", "col"], data["strain"]),
    ),
)

list_datasets.append(ds)

plot.plot_stress_subplot(
    list_datasets,
    cfg["uniaxial"]["output_directory"] + "stress.png",
    "Stress",
)
