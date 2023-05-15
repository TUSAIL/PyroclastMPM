# %%
from pyroclastmpm import (
    NoSlipWall,
    LinearElastic,
    NewtonFluid,
    ParticlesContainer,
    NodesContainer,
    USL,
    MUSL,
    APIC,
    SlipWall,
    LinearShapeFunction,
    CubicShapeFunction,
    set_globals,
    Gravity,
    VTK
)

import numpy as np
import matplotlib.pyplot as plt


# %%
def create_circle(center: np.array, radius: float, cell_size: float, ppc: int = 1):
    """Create a 2D circle

    :param center: center of circle
    :param radius: radius of circle
    :param cell_size: cell size of background grid
    :param ppc: particles per cell, defaults to 1
    """
    start = center - radius
    end = center + radius
    spacing = cell_size / ppc
    tol = +0.002  # prevents points

    x = np.arange(start[0], end[0] + spacing, spacing)
    y = np.arange(start[1], end[1] + spacing, spacing)
    z = np.zeros(len(x))
    xv, yv, zv = np.meshgrid(x, y, z)
    grid_coords = np.array(list(zip(xv.flatten(), yv.flatten(), zv.flatten()))).astype(
        np.float64
    )

    circle_mask = (grid_coords[:, 0] - center[0]) ** 2 + (
        grid_coords[:, 1] - center[1]
    ) ** 2 < radius**2 + tol

    return grid_coords[circle_mask]


# %%
domain_start = np.array([0.0, 0.0, 0.0])
domain_end = np.array([1.0, 1.0, 1.0])
cell_size = 1 / 40
ppc = 2

rho0 = 10000

# Define global simulation parameters
set_globals(
    dimension=2,
    dt=0.001,
    shape_function=CubicShapeFunction,
    output_directory="./output",
    out_type=VTK
)


# %%

nodes = NodesContainer(
    node_start=domain_start, node_end=domain_end, node_spacing=cell_size
)

print(f"Total number of cells: {nodes.num_nodes_total}")


# %%
# setup particles

positions = create_circle(
    center=np.array([0.5, 0.5, 0.0]), radius=0.15, cell_size=cell_size, ppc=ppc
)


particles = ParticlesContainer(
    positions=positions,
)

material = LinearElastic(density=rho0,E=1000, pois=0.3)

wallx0 = SlipWall(wallplane="x0")
wallx1 = SlipWall(wallplane="x1")
wally0 = SlipWall(wallplane="y0")
wally1 = SlipWall(wallplane="y1")


gravity = Gravity(gravity=np.array([0, -0.1, 0]))

MPM = USL(
    particles=particles,
    nodes=nodes,
    materials=[material],
    boundaryconditions=[gravity,wallx0,wallx1,wally0,wally1],
    total_steps=8000,
    output_steps=400,
    output_start=0
)
MPM.run()
