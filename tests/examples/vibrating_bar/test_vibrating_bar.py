import os
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyroclastmpm import (
    LinearElastic,
    ParticlesContainer,
    NodesContainer,
    USL, TLMPM, MUSL,
    LinearShapeFunction,
    set_globals,
    global_dimension,
    BodyForce,
    VTK, CSV
)

# Functions to test
# [x] vibrating bar with USL
# [x] vibrating bar with MUSL
# [x] vibrating bar with TLMPM



@pytest.mark.parametrize("solver_type", [("usl"), ("musl"), ("tlmpm")])
def test_vibratingbar_usl(solver_type):

    if global_dimension != 1: return
    
    if solver_type == "usl":
        solverclass = USL
        output_directory = os.path.dirname(__file__) + "/output_usl/"
        plot_directory = os.path.dirname(__file__) + "/plots_usl/"
    elif solver_type == "musl":
        solverclass = MUSL
        output_directory = os.path.dirname(__file__) + "/output_musl/"
        plot_directory = os.path.dirname(__file__) + "/plots_musl/"
    elif solver_type == "tlmpm":
        solverclass = TLMPM
        output_directory = os.path.dirname(__file__) + "/output_tlmpm/"
        plot_directory = os.path.dirname(__file__) + "/plots_tlmpm/"
    else:
        assert False, "Invalid solver type"

    L = 25.
    E = 100
    cell_size = L / 49
    rho = 1
    c = np.sqrt(E / rho)  # critical wave speed
    delta_t = 0.1 * cell_size / c  # time step

    mode = 1  # mode of vibration
    v0 = 0.1  # amplitude of vibration
    beta0 = ((2 * mode - 1) / 2.0) * (np.pi / L)  # wave number

    total_time = 50
    
    ppc_1d = 1

    set_globals(
        dt=delta_t,
        particles_per_cell=ppc_1d,
        shape_function=LinearShapeFunction,
        output_directory=output_directory
    )

    nodes = NodesContainer(
        node_start=[0.],
        node_end=[L],
        node_spacing=cell_size,
        output_formats=[VTK])
    

    # exit()
    mp_coords = np.arange(0, L, cell_size/ppc_1d) + cell_size/(2.*ppc_1d) 

    mp_coords = np.expand_dims(mp_coords, axis=1)

    # print("tot nodes",nodes.num_nodes_total,np.shape(mp_coords))
    # exit()
    # plt.figure(figsize=(20,10))
    # plt.plot(mp_coords,np.zeros_like(mp_coords),"o")
    # plt.plot(nodes.give_coords(),np.zeros_like(nodes.give_coords()),"x")
    # plt.savefig(plot_directory + "/bar.png")
    mp_vels = v0 * np.sin(beta0 * mp_coords)

    particles = ParticlesContainer(
        positions=mp_coords, velocities=mp_vels, output_formats=[VTK, CSV])

    node_coords = nodes.give_coords()

    # bar is fixed in one end and free in another
    mask_body_force = np.zeros(node_coords.shape[0], dtype=bool)
    mask_body_force[0] = 1
    bodyforce = BodyForce(mode="fixed", values=np.zeros(
        node_coords.shape), mask=mask_body_force)

    material = LinearElastic(density=rho, E=E, pois=0)

    MPM = solverclass(
        particles=particles,
        nodes=nodes,
        materials=[material],
        boundaryconditions=[bodyforce],
        total_steps=int(total_time / delta_t),  # 3 seconds
        output_steps=20,
        output_start=0,
        alpha=0.99
    )

    MPM.run()

    vcom_num = []  # center of mass velocities (numerical)
    t_num = []  # time intervals (numerical)
    for i in range(MPM.output_start, MPM.total_steps, MPM.output_steps):
        df = pd.read_csv(output_directory + 'particles{}.csv'.format(i))
        v_com = (df["Velocity"]*df["Mass"]).sum()/df["Mass"].sum()
        vcom_num.append(v_com)
        t_num.append(delta_t*i)

    t_exact = np.arange(0, total_time, delta_t*20)
    omegan = beta0*np.sqrt(E/rho)
    vcom_exact = np.cos(omegan*t_exact)*v0/(beta0*L)

    mse = ((vcom_exact - vcom_num)**2).mean(axis=0)

    plt.plot(t_num, vcom_num, label="numerical")
    plt.plot(t_exact, vcom_exact, label="exact")
    plt.xlabel("Time (s)")
    plt.ylabel("Center of mass velocity (m/s)")
    plt.legend(["Numerical", "Exact"])
    plt.savefig(plot_directory + "/vibrating_bar.png")
    plt.clf()
    assert mse < 1e-3, "Mean Squared error too high"
