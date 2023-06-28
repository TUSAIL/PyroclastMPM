# Loading a config file and running a uniaxial stress test
import tomllib

import numpy as np
import plot
import xarray as xr
from constitutive_analysis.loading import strain_controlled
from pyroclastmpm.MPM3D import (
    MohrCoulomb,
    ParticlesContainer,
    VonMises,
    set_global_timestep,
)

# load config file
with open("./config.toml", "rb") as f:
    cfg = tomllib.load(f)

# Time step for increment of deformation gradient
dt = cfg["global"]["timestep"]
set_global_timestep(dt)

# Create particles and initialize material
particle_vm = ParticlesContainer([[0.0, 0.0, 0.0]])
particle_mc = ParticlesContainer([[0.0, 0.0, 0.0]])

material_vm = VonMises(
    cfg["von_mises"]["density"],
    cfg["von_mises"]["E"],
    cfg["von_mises"]["pois"],
    cfg["von_mises"]["yield_stress"],
    cfg["von_mises"]["H"],
)

material_mc = MohrCoulomb(
    cfg["mohr_coulomb"]["density"],
    cfg["mohr_coulomb"]["E"],
    cfg["mohr_coulomb"]["pois"],
    cfg["mohr_coulomb"]["cohesion"],
    cfg["mohr_coulomb"]["friction_angle"],
    cfg["mohr_coulomb"]["dilatancy_angle"],
    cfg["mohr_coulomb"]["H"],
)
particle_vm, _ = material_vm.initialize(particle_vm, 0)
particle_mc, _ = material_mc.initialize(particle_mc, 0)

deps = np.zeros((3, 3))

deps[0, 0] = np.array(cfg["uniaxial"]["deps_xx"]) * dt


data_mc = {"stress": [], "strain": [], "eps_e": []}
data_vm = {"stress": [], "strain": [], "eps_e": []}


def callback_vm(particles, _, step):
    global data_vm
    if step % cfg["global"]["output_steps"] == 0:
        data_vm["stress"].append(particles.stresses[0])
        F = np.array(particles.F[0])
        strain = 0.5 * (F.T + F) - np.identity(3)
        data_vm["strain"].append(strain)


def callback_mc(particles, _, step):
    global data_mc
    if step % cfg["global"]["output_steps"] == 0:
        data_mc["stress"].append(particles.stresses[0])
        F = np.array(particles.F[0])
        strain = 0.5 * (F.T + F) - np.identity(3)
        data_mc["strain"].append(strain)


list_datasets = []

# run uniaxial stress test
particle_vm, material_vm = strain_controlled(
    particle_vm,
    material_vm,
    deps * dt,
    cfg["global"]["num_steps"],
    dt,
    callback_vm,
)


ds = xr.Dataset(
    data_vars=dict(
        stress=(["step", "row", "col"], data_vm["stress"]),
        strain=(["step", "row", "col"], data_vm["strain"]),
    ),
)

# list_datasets.append(ds)

# run uniaxial stress test
particles_mc, material_mc = strain_controlled(
    particle_mc,
    material_mc,
    deps * dt,
    cfg["global"]["num_steps"],
    dt,
    callback_mc,
)

ds = xr.Dataset(
    data_vars=dict(
        stress=(["step", "row", "col"], data_mc["stress"]),
        strain=(["step", "row", "col"], data_mc["strain"]),
    ),
)

list_datasets.append(ds)

plot.plot_stress_subplot(
    list_datasets,
    cfg["uniaxial"]["output_directory"] + "stress.png",
    "Stress",
)
