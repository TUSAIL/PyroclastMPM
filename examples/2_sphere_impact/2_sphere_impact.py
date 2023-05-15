# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

print(f"Running a {global_dimension}D simulation")

set_globals(
    dt=0.001,
    particles_per_cell=4,
    shape_function=LinearShapeFunction,
    output_directory="./output"
)

# %%


def create_circle(center: np.array, radius: float, cell_size: float, ppc: int = 1):
    start, end = center-radius, center+radius
    spacing = cell_size/ppc
    tol = +0.00005  # prevents points
    x = np.arange(start[0], end[0] + spacing, spacing) + 0.5*spacing
    y = np.arange(start[1], end[1] + spacing, spacing) + 0.5*spacing
    xv, yv = np.meshgrid(x, y)
    grid_coords = np.array(
        list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)
    circle_mask = (grid_coords[:, 0] - center[0]) ** 2 + (
        grid_coords[:, 1] - center[1]
    ) ** 2 < radius**2 + tol
    return grid_coords[circle_mask]


# %%
domain_start, domain_end = np.array([[0., 0.], [1., 1.]])
cell_size = 1./20
ppc = 2
# %%
nodes = NodesContainer(node_start=domain_start,
                       node_end=domain_end, node_spacing=cell_size)

print(f"Total number of cells: {nodes.num_nodes_total}")

node_coords = nodes.give_coords()

# %%

cicrle_centers = np.array([[0.2, 0.2], [0.8, 0.8]])
circles = np.array([
    create_circle(
        center=center,
        radius=0.2,
        cell_size=cell_size,
        ppc=ppc) for center in cicrle_centers
])

# %%
plt.scatter(node_coords[:, 0], node_coords[:, 1])
for circle in circles:
    plt.scatter(circle[:, 0], circle[:, 1])
# %%

positions = np.vstack(circles)

velocities1 = np.ones(circles[0].shape) * 0.1
velocities2 = np.ones(circles[1].shape) * -0.1
velocities = np.vstack((velocities1, velocities2))

color1 = np.zeros(len(circles[0]))
color2 = np.ones(len(circles[1]))
colors = np.concatenate([color1, color2]).astype(int)

# %%
mat0 = LinearElastic(density=1000, E=1000, pois=0.3)

particles = ParticlesContainer(
    positions=positions,
    velocities=velocities,
    colors=colors,
    output_formats=[VTK, CSV])


print(f"Total number of particles {particles.num_particles}")

# %%
MPM = USL(
    particles=particles,
    nodes=nodes,
    materials=[mat0, mat0],
    total_steps=3600,  # 3 seconds
    output_steps=100,
    output_start=0,
)
#%%
MPM.run()


# %%
Kinetic_energy = []
time = []
for step in range(0, 3600, 100):
    time.append(step*0.001)
    df = pd.read_csv(
        f'./output/particles{step}.csv', delimiter=',')
    
    df["KE"] = 0.5*df["Mass"]*(df["Velocity:0"]**2+df["Velocity:1"]**2)
    # vel = df[["Velocity:0","Velocity:1"]].values
    # mass = 
    # Ek = 0.5*np.einsum("ij,ij->i",vel,vel)@df["Mass"].values[:,None]
   
    Kinetic_energy.append(np.sum( df["KE"].sum()))

# print(vel@vel)

# %%
import matplotlib.pyplot as plt
plt.plot(time,Kinetic_energy,label="Circle 1")
# %%



# %%
