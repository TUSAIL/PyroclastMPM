# %%
"""
Water drop example
"""


import matplotlib.pyplot as plt
import numpy as np
import pyroclastmpm.MPM2D as pm

# material parameters
rho = 997.5
bulk_modulus = 2.0 * 10**6
mu = 0.001  # slightly lower than normal water
g = -10.0


# background grid
origin, end = [0.0, 0.0], [0.9, 0.9]

cell_size = 1.0 / 80
c = np.sqrt(bulk_modulus / rho)
dt = 0.1 * cell_size / c  # dt is 2.791588929265912e-05
output_formats = ["vtk"]

# global

particles_per_cell = 1
shape_function = "cubic"
output_directory = "output"
total_steps = 80000  # total time is about 2.23 seconds
output_steps = 1000  # output every 0.0001395794464632487 seconds

# solver
alpha = 0.993


pm.set_globals(
    dt,
    particles_per_cell,
    shape_function,
    output_directory,
)

nodes = pm.NodesContainer(origin, end, cell_size)

nodes.set_output_formats(["vtk"])


node_coords = np.array(nodes.give_coords()).astype(float)

pnts = np.random.uniform((0.55, 0.2), (0.65, 0.5), size=(2400, 2))

vels = np.zeros_like(pnts)
vels[:, 1] = 0
print(pnts.shape)


plt.scatter(pnts[:, 0], pnts[:, 1], s=0.5)
plt.scatter(node_coords[:, 0], node_coords[:, 1], s=0.5)

particles = pm.ParticlesContainer(pnts, vels)
particles.set_output_formats(output_formats)

# 4. Create material
material = pm.NewtonFluid(rho, mu, bulk_modulus, 7)

# 5. Create boundary conditions
gravity_bc = pm.Gravity([0, g])

domain_bc = pm.NodeDomain(face0_mode=[1, 1], face1_mode=[1, 1])

# # # 5. Create solver and run
MPM = pm.USL(
    particles, nodes, [material], [gravity_bc, domain_bc], alpha=alpha
)

MPM.run(total_steps, output_steps)

# %%
