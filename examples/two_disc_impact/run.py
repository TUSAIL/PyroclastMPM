"""This example shows two spheres impacting each other [1].

[1] Sulsky, Deborah, Zhen Chen, and Howard L. Schreyer.
"A particle method for history-dependent materials."
Computer methods in applied mechanics and engineering 118.1-2 (1994): 179-196.
"""


import numpy as np
import pyroclastmpm.MPM2D as pm

# 1. Load config file and set global variables

# global
dt = 0.001
particles_per_cell = 4
shape_function = "linear"
output_directory = "output"
total_steps, output_steps, output_start = 3000, 100, 0


# nodes
origin, end = [0.0, 0.0], [1.0, 1.0]
cell_size = 0.05
output_formats = ["vtk"]

# particles
circle1_center = np.array([0.255, 0.255])
circle2_center = np.array([0.745, 0.745])
circle_radius = 0.2
output_formats = ["vtk"]

# material
density = 1000
E = 1000
pois = 0.3

# solver
alpha = 1.0  # pure flip

pm.set_globals(
    dt,
    particles_per_cell,
    shape_function,
    output_directory,
)


def create_circle(
    center: np.array, radius: float, cell_size: float, ppc: int = 2
):
    """Generate a circle of particles.

    Args:
        center (np.array): center of the circle
        radius (float): radius of the circle
        cell_size (float): size of the background grid cells
        ppc (int, optional): particles per cell. Defaults to 2.

    Returns:
        np.array: coordinates of the particles
    """
    start, end = center - radius, center + radius
    spacing = cell_size / (ppc / 2)
    tol = +0.00005  # Add a tolerance to avoid numerical issues
    x = np.arange(start[0], end[0] + spacing, spacing) + 0.5 * spacing
    y = np.arange(start[1], end[1] + spacing, spacing) + 0.5 * spacing
    xv, yv = np.meshgrid(x, y)
    grid_coords = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(
        np.float64
    )
    circle_mask = (grid_coords[:, 0] - center[0]) ** 2 + (
        grid_coords[:, 1] - center[1]
    ) ** 2 < radius**2 + tol
    return grid_coords[circle_mask]


# 2. Create background grid
nodes = pm.NodesContainer(origin, end, cell_size)

nodes.set_output_formats(output_formats)

# 3. Create particles using the circles module (in same folder)
circle_centers = np.array([circle1_center, circle2_center])

# list of two circles
circles = np.array(
    [
        create_circle(center, circle_radius, cell_size, particles_per_cell)
        for center in circle_centers
    ]
)


# concatenate the two circles into a single array
pos = np.vstack(circles)

# the spheres are moving towards each other
vel1 = np.ones(circles[0].shape) * 0.1
vel2 = np.ones(circles[1].shape) * -0.1
vels = np.vstack((vel1, vel2))

particles = pm.ParticlesContainer(pos, vels)
particles.set_output_formats(output_formats)

# 4. Create material
material = pm.LinearElastic(density, E, pois)

# # # 5. Create solver and run
MPM = pm.USL(particles, nodes, [material], alpha=alpha)

MPM.run(total_steps, output_steps)
