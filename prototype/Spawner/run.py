# %%
import tomllib

import numpy as np
from pyroclastmpm import (  # LocalGranularRheology,
    USL,
    VTK,
    CubicShapeFunction,
    Gravity,
    NewtonFluid,
    NodeDomain,
    NodesContainer,
    ParticlesContainer,
    RigidBodyLevelSet,
    get_bounds,
    global_dimension,
    grid_points_in_volume,
    grid_points_on_surface,
    set_globals,
)

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

# check if code is compiled for correct dimension
if global_dimension != config["global"]["dimension"]:
    raise ValueError(
        f"This example only works in {config['global']['dimension']}D. The code is compiled for {global_dimension}D."
    )

# load stls
domain_start, domain_end = get_bounds(config["project"]["domain_file"])

# calculate cell size
cell_size = abs(np.min(domain_end - domain_start) / config["global"]["cell_size_ratio"])


c = np.sqrt(config["material"]["bulk_modulus"] / config["material"]["density"])
dt = 0.1 * cell_size / c

print(dt)
material_coords = np.array(
    grid_points_in_volume(
        config["project"]["material_file"],
        cell_size,
        config["global"]["particles_per_cell"],
    )
)


particle_positions = material_coords

print(f"Num particles {particle_positions.shape}.")
# %%
set_globals(
    dt=dt,
    particles_per_cell=config["global"]["particles_per_cell"],
    shape_function=CubicShapeFunction,
    output_directory=config["global"]["output_directory"],
)

# create nodes
nodes = NodesContainer(
    node_start=domain_start, node_end=domain_end, node_spacing=cell_size
)


# create particles
particle_velocites = np.zeros(particle_positions.shape)

particle_velocites[:, 1] = -1.0


particles = ParticlesContainer(
    positions=particle_positions,
    velocities=particle_velocites,
    output_formats=[VTK],
)

# particles.set_spawner(spawn_rate, spawn_volume)

print(f"num_p {particles.num_particles}, num_c {nodes.num_nodes_total} \n")

domain = NodeDomain(
    axis0_mode=config["boundaryconditions"]["axis0_mode"],
    axis1_mode=config["boundaryconditions"]["axis1_mode"],
)


wallgrav = Gravity(
    gravity=config["boundaryconditions"]["gravity"],
)


material = NewtonFluid(
    config["material"]["density"],
    config["material"]["mu"],
    config["material"]["bulk_modulus"],
    7.0,
)

MPM = USL(
    particles=particles,
    nodes=nodes,
    materials=[material],
    total_steps=config["global"]["total_steps"],
    output_steps=config["global"]["output_steps"],
    output_start=config["global"]["output_start"],
    boundaryconditions=[domain, wallgrav],
)

MPM.run()
