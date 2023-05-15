#%%
from pyroclastmpm import (
    LinearElastic,
    ParticlesContainer,
    NodesContainer,
    USL,
    LinearShapeFunction,
    BodyForce,
    set_globals,
    CSV,
)

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


import numpy as np
import matplotlib.pyplot as plt

L = 25.  # length of the bar
cell_size = 25 / 14.0  # 14 cells
ppc = 1  # particles per cell
rho = 1  # density of the bar
E = 100  # Young's modulus of the bar
pois = 0  # Poisson's ratio of the bar
c = np.sqrt(E / rho)  # critical wave speed
delta_t = 0.1 * cell_size / c  # time step

mode = 0  # mode of vibration
v0 = 0.1  # amplitude of vibration
beta0 = ((2 * mode - 1) / 2.0) * (np.pi / L)  # wave number

total_time = 50

total_steps = int(total_time / delta_t)

outstep = 10

# # Define global simulation parameters
set_globals(
    dimension=1,
    dt=delta_t,
    shape_function=LinearShapeFunction,
    output_directory=dir_path+"/output",
    out_type=CSV,
)

nodes = NodesContainer(
    node_start=np.array([0, 0.0, 0.0]),
    node_end=np.array([L, 0.0, 0.0]),
    node_spacing=cell_size,
)

node_coords = nodes.give_coords()

mp_coords = np.arange(0, L, cell_size / ppc) + cell_size / ppc / 2.0
zero_padding = np.zeros((len(mp_coords), 2))
mp_coords = np.hstack((mp_coords[..., np.newaxis], zero_padding))

mp_vels = np.zeros(mp_coords.shape)
mp_vels[:, 0] = v0 * np.sin(beta0 * mp_coords[:, 0])

mat = LinearElastic(density=rho, E=E, pois=0)

particles = ParticlesContainer(positions=mp_coords, velocities=mp_vels)

# bar is fixed in one end and free in another
mask_body_force = np.zeros(node_coords.shape[0], dtype=bool)
mask_body_force[0]= 1
bodyForce = BodyForce(mode="fixed", values=np.zeros(node_coords.shape), mask=mask_body_force)

MPM = USL(
    particles=particles,
    nodes=nodes,
    materials=[mat],
    boundaryconditions=[bodyForce],
    total_steps=total_steps,  # 3 seconds
    output_steps=outstep,
    output_start=0,
)

MPM.run()

import pandas as pd
import matplotlib.pyplot as plt

vcom_num = [] # center of mass velocities (numerical)
t_num =[] # time intervals (numerical)
for i in range(MPM.output_start,MPM.total_steps,MPM.output_steps):
    df = pd.read_csv(dir_path+ '/output/particles_0_{}.csv'.format(i))
    v_com = (df["Velocity:0"]*df["Mass"]).sum()/df["Mass"].sum()
    vcom_num.append(v_com)
    t_num.append(delta_t*i)

t_exact = np.arange(0, total_time, 0.1)
omegan = beta0*np.sqrt(E/rho)
vcom_exact = np.cos(omegan*t_exact)*v0/(beta0*L)


plt.plot(t_num,vcom_num)
plt.plot(t_exact,vcom_exact)
plt.xlabel("Time (s)")
plt.ylabel("Center of mass velocity (m/s)")
ply.legend(["Numerical","Exact"])

plt.show()


# %%
