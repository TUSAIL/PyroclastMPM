import numpy as np
import pyroclastmpm.MPM2D as pm

# 1. Load config file and set global variables

# dam
dam_height = 2.0
dam_length = 4.0

# material parameters
rho = 997.5
bulk_modulus = 2.0 * 10**6
mu = 0.001
g = -9.81


# background grid
origin, end = [0.0, 0.0], [6.0, 6.0]


cell_size = 6.0 / 60
# timestep
c = np.sqrt(bulk_modulus / rho)
dt = 0.1 * cell_size / c

# print(dt, cell_size, c)

output_formats = ["vtk"]


# global
particles_per_cell = 2  # 1d?
shape_function = "cubic"
output_directory = "output"
total_time = 1.6  # seconds
total_steps = 80000
output_steps = 1000


alpha = 0.99


pm.set_globals(
    dt,
    particles_per_cell,
    shape_function,
    output_directory,
)

nodes = pm.NodesContainer(origin, end, cell_size)

nodes.set_output_formats(output_formats)


node_coords = np.array(nodes.give_coords()).astype(float)


# Create particle positions

sep = cell_size / 2
x = np.arange(0, dam_length + sep, sep) + 1.5 * sep
y = np.arange(0, dam_height + sep, sep) + 2.5 * sep
xv, yv = np.meshgrid(x, y)
pnts = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)


print(pnts.shape, node_coords.shape)
particles = pm.ParticlesContainer(pnts)
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
