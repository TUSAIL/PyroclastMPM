# %%
import numpy as np
import tomllib

from pyroclastmpm import (
    LinearElastic,
    ParticlesContainer,
    NodesContainer,
    USL, 
    LinearShapeFunction,
    set_globals,
    global_dimension,
    VTK, CSV
)

# project specific
from circles import create_circle

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

# check if code is compiled for correct dimension
if global_dimension != config['global']['dimension']:
    raise ValueError("This example only works in 2D")

# set global variables

set_globals(
    dt=config['global']['dt'],
    particles_per_cell=config['global']['particles_per_cell'],
    shape_function=LinearShapeFunction,
    output_directory=config['global']['output_directory']
)

# create nodes
nodes = NodesContainer(
    node_start=config['nodes']['node_start'],
    node_end=config['nodes']['node_end'],
    node_spacing=config['nodes']['node_spacing'],
    output_formats=[VTK, CSV]
)

circle_centers = np.array([
    np.array(config['particles']['circle1_center']),
    np.array(config['particles']['circle2_center'])
])

circles = np.array([
    create_circle(
        center=center,
        radius=config['particles']['circle_radius'],
        cell_size=config['nodes']['node_spacing'],
        ppc_1d=config['global']['particles_per_cell']/2  # special case for 2D
        ) for center in circle_centers
])

positions = np.vstack(circles)

velocities1 = np.ones(circles[0].shape) * 0.1
velocities2 = np.ones(circles[1].shape) * -0.1
velocities = np.vstack((velocities1, velocities2))

color1 = np.zeros(len(circles[0]))
color2 = np.ones(len(circles[1]))
colors = np.concatenate([color1, color2]).astype(int)

particles = ParticlesContainer(
    positions=positions,
    velocities=velocities,
    colors=colors,
    output_formats=[VTK, CSV]
    )


material = LinearElastic(
    density=config['material']['density'],
    E=config['material']['E'],
    pois=config['material']['pois'])

MPM = USL(
    particles=particles,
    nodes=nodes,
    materials=[material, material],
    alpha=config['solver']['alpha'],
    total_steps=config['global']['total_steps'],  # 3 seconds
    output_steps=config['global']['output_steps'],
    output_start=config['global']['output_start']
)

MPM.run()





# # create boundary conditions
# grav = Gravity(gravity=config['boundaryconditions']['gravity'])

# domain = PlanarDomain(
#     axis0_friction=config['boundaryconditions']['axis0_friction'],
#     axis1_friction=config['boundaryconditions']['axis1_friction']
# )

# # domain = NodeDomain(
# #    axis0_mode=[1,1],axis1_mode=[1,1]
# #    )

# MPM = MUSL(
#     particles=particles,
#     nodes=nodes,
#     materials=[material],
#     boundaryconditions=[domain, grav],
#     alpha=config['solver']['alpha'],
#     total_steps=config['global']['total_steps'],  # 3 seconds
#     output_steps=config['global']['output_steps'],
#     output_start=config['global']['output_start']
# )

# MPM.run()
# %%
