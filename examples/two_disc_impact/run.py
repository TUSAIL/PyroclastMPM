"""_summary_

This example shows two spheres impacting each other [1].


How to run:
    - Run `make` in this folder
    - Alternatively, run `python3 run.py` but make sure the output dictory is created.

Requirements:
    - pyroclastmpm (2D compiled), PyVista, numpy, toml, circles

Notes:
 - The spheres are created using the `circles` module
 - Data is stored in CSV and VTK format in the `output` folder
 - Simulation parameters are loaded from a TOML file called `config.toml`
 - Postprocessing is done using PyVista in `postprocess.py`.

[1] Sulsky, Deborah, Zhen Chen, and Howard L. Schreyer.
"A particle method for history-dependent materials."
Computer methods in applied mechanics and engineering 118.1-2 (1994): 179-196.
"""

import tomli

import numpy as np
from circles import create_circle
from pyroclastmpm import (
    USL,
    LinearElastic,
    NodesContainer,
    ParticlesContainer,
    check_dimension,
    set_globals,
)

# 1. Load config file and set global variables
with open("./config.toml", "rb") as f:
    config = tomli.load(f)

check_dimension(config["global"]["dimension"])

set_globals(
    dt=config["global"]["dt"],
    particles_per_cell=config["global"]["particles_per_cell"],
    shape_function=config["global"]["shape_function"],
    output_directory=config["global"]["output_directory"],
)


# 2. Create background grid
nodes = NodesContainer(
    node_start=config["nodes"]["node_start"],
    node_end=config["nodes"]["node_end"],
    node_spacing=config["nodes"]["node_spacing"],
)

nodes.set_output_formats(config["nodes"]["output_formats"])

# 3. Create particles using the circles module (in same folder)
circle_centers = np.array(
    [
        np.array(config["particles"]["circle1_center"]),
        np.array(config["particles"]["circle2_center"]),
    ]
)

# list of two circles
circles = np.array(
    [
        create_circle(
            center=center,
            radius=config["particles"]["circle_radius"],
            cell_size=config["nodes"]["node_spacing"],
            ppc=config["global"]["particles_per_cell"],
        )
        for center in circle_centers
    ]
)


# concatenate the two circles into a single array
positions = np.vstack(circles)
print(positions.shape)
# the spheres are moving towards each other
velocities1 = np.ones(circles[0].shape) * 0.1
velocities2 = np.ones(circles[1].shape) * -0.1
velocities = np.vstack((velocities1, velocities2))

# for convenience, we assign a color to each sphere
# (this is only used for visualization)
# material remains the same
color1 = np.zeros(len(circles[0]))
color2 = np.ones(len(circles[1]))
colors = np.concatenate([color1, color2]).astype(int)

particles = ParticlesContainer(
    positions=positions, velocities=velocities, colors=colors
)
particles.set_output_formats(config["particles"]["output_formats"])

# 4. Create material
material = LinearElastic(
    density=config["material"]["density"],
    E=config["material"]["E"],
    pois=config["material"]["pois"],
)

# 5. Create solver and run
MPM = USL(
    particles=particles,
    nodes=nodes,
    materials=[material, material],
    alpha=config["solver"]["alpha"],
    total_steps=config["global"]["total_steps"],
    output_steps=config["global"]["output_steps"],
    output_start=config["global"]["output_start"],
)

MPM.run()
