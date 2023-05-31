# %%
from pyroclastmpm import (
    Gravity,
    NodeDomain,
    ParticlesContainer,
    NodesContainer,
    USL,
    # MUSL,
    CubicShapeFunction,
    LinearShapeFunction,
    set_globals,
    get_bounds,
    grid_points_in_volume,
    grid_points_on_surface,
    RigidBodyLevelSet,
    LocalGranularRheology,
    VTK,CSV,GTFL,
    global_dimension
)

import tomllib
import numpy as np
import matplotlib.pyplot as plt

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

# check if code is compiled for correct dimension
if global_dimension != config['global']['dimension']:
    raise ValueError(f"This example only works in {config['global']['dimension']}D. The code is compiled for {global_dimension}D.")

# load stls
domain_start, domain_end = get_bounds(config['project']['domain_file'])

# calculate cell size
cell_size = abs(np.min(domain_end - domain_start) / config['global']['cell_size_ratio'])

material_coords = np.array(
    grid_points_in_volume(config['project']['material_file'], cell_size, config['global']['particles_per_cell']))

rigid_coords, normals = grid_points_on_surface(
    config['project']['scoop_file'], cell_size=cell_size, point_per_cell=1
)


rigid_mask = np.concatenate((
    np.zeros(np.shape(material_coords)[0], dtype=bool),
    np.ones(np.shape(rigid_coords)[0], dtype=bool)
    ))


particle_positions = np.concatenate((material_coords, rigid_coords), axis=0)

print(len(particle_positions), len(rigid_mask))
#%%
set_globals(
    dt=config['global']['dt'],
    particles_per_cell=config['global']['particles_per_cell'],
    shape_function=CubicShapeFunction,
    output_directory=config['global']['output_directory']
)

# create nodes
nodes = NodesContainer(
    node_start=domain_start,
    node_end=domain_end,
    node_spacing=cell_size
)




# create particles
particle_velocites = np.zeros(particle_positions.shape)

particles = ParticlesContainer(
    positions=particle_positions, 
    velocities=particle_velocites,
    is_rigid=rigid_mask,
    output_formats=[VTK]
    )

print(f"num_p {particles.num_particles}, num_c {nodes.num_nodes_total} \n")




frames, lx, ly, lz, rx, ry, rz = np.loadtxt(config['project']['scoop_motion_file']).T
locations = np.vstack([lx, ly, lz]).T
rotations = np.vstack([rx, ry, rz]).T

rigidboundary = RigidBodyLevelSet(
    frames=frames.astype(int),
    locations=locations,
    rotations=rotations,
    output_formats=[GTFL]
)

domain = NodeDomain(
   axis0_mode=config['boundaryconditions']['axis0_mode'],
   axis1_mode=config['boundaryconditions']['axis1_mode'],
   )


wallgrav = Gravity(gravity=config['boundaryconditions']['gravity'],)



material = LocalGranularRheology(
    density=config['material']['density'],
    E=config['material']['E'],
    pois=config['material']['pois'],
    I0=config['material']['I0'],
    mu_s=config['material']['mu_s'],
    mu_2=config['material']['mu_2'],
    rho_c=config['material']['rho_c'],
    particle_diameter=config['material']['particle_diameter'],
    particle_density=config['material']['particle_density'],
)
MPM = USL(
    particles=particles,
    nodes=nodes,
    materials=[material],
    total_steps=config['global']['total_steps'],
    output_steps=config['global']['output_steps'],
    output_start=config['global']['output_start'],
    boundaryconditions=[
        domain,
        wallgrav,
        rigidboundary,
    ],
)

MPM.run()

# %%
