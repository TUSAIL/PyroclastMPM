# BSD 3-Clause License
# Copyright (c) 2023, Retief Lubbe
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this
#  list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pyroclastmpm import (
    CSV,
    MUSL,
    USL,
    VTK,
    LinearElastic,
    LinearShapeFunction,
    NodesContainer,
    ParticlesContainer,
    global_dimension,
    set_globals,
)


def create_circle(
    center: np.array, radius: float, cell_size: float, ppc_1d: int = 1
):
    start, end = center - radius, center + radius
    spacing = cell_size / ppc_1d
    tol = +0.00005  # prevents points
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


@pytest.mark.parametrize("solver_type", [("usl"), ("musl")])
def test_two_disk_impact(solver_type):
    if global_dimension != 2:
        return
    if solver_type == "usl":
        solverclass = USL
        output_directory = os.path.dirname(__file__) + "/output_usl/"
        plot_directory = os.path.dirname(__file__) + "/plots_usl/"
    elif solver_type == "musl":
        solverclass = MUSL
        output_directory = os.path.dirname(__file__) + "/output_musl/"
        plot_directory = os.path.dirname(__file__) + "/plots_musl/"
    else:
        assert False, "Invalid solver type"

    set_globals(
        dt=0.001,
        particles_per_cell=4,
        shape_function=LinearShapeFunction,
        output_directory=output_directory,
    )

    nodes = NodesContainer(
        node_start=[0.0, 0.0], node_end=[1.0, 1.0], node_spacing=1.0 / 20
    )

    circle_centers = np.array([[0.2, 0.2], [0.8, 0.8]])

    circles = np.array(
        [
            create_circle(
                center=center, radius=0.2, cell_size=1 / 20, ppc_1d=2
            )
            for center in circle_centers
        ]
    )

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
        output_formats=[VTK, CSV],
    )

    material = LinearElastic(density=1000, E=1000, pois=0.3)

    MPM = solverclass(
        particles=particles,
        nodes=nodes,
        materials=[material, material],
        alpha=0.99,
        total_steps=3600,  # 3 seconds
        output_steps=100,
        output_start=0,
    )

    MPM.run()

    KE = []

    time = []
    for step in range(0, 3600, 100):
        time.append(step * 0.001)
        df = pd.read_csv(
            output_directory + f"/particles{step}.csv", delimiter=","
        )

        df["KE"] = (
            0.5 * df["Mass"] * (df["Velocity:0"] ** 2 + df["Velocity:1"] ** 2)
        )

        KE.append(np.sum(df["KE"].sum()))

    np.testing.assert_approx_equal(KE[0], 2.51, 0.1)

    error = abs((KE[-1] - KE[0]) / KE[0])

    plt.plot(time, KE)
    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic Energy (J)")

    plt.savefig(plot_directory + "/KE.png")

    assert error < 0.2, "KE should be conserved"
