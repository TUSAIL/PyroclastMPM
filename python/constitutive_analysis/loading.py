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


import numpy as np


def servo_control(
    particles,
    material,
    max_load_rate,
    control,
    targets,
    tol=1e-3,
    acceleration=1,
    constant_load=False,
    callback=None,
    dim=3,
):
    """Servo control loading:

    There are two modes of operation:
        - Strain controlled
        - Stress controlled


    In controlled mode, the strain rate is controlled to achieve a target stress/strain.
    Strain control has an option to keep the load constant util loading is complete.

    The servo control algorithm is useful for triaxial or isotropic compression tests.

    Args:
        particles (ParticlesContainer): PyroclastMPM ParticlesContainer
        material (Material): PyroclastMPM material
        max_load_rate (float): Maximum loading rate (when acceleration = 1) deps*dt
        control (list): Type of control (stress or strain) for each dimension.
                          E.g. ["stress", "strain", "stress"]
        targets (list): Target stress/strain for each dimension.
                          E.g. [-1000, -0.2, -1000]
        tol (float, optional): Tolarance criteria to accept targets.
                                Defaults to 1e-3.
        acceleration (float, optional): Acceleration of the servo controller.
                                      e.g 2,  Defaults to 1.
        constant_load (bool, optional): If strain controlled uses a constant load.
                                        Then ma_load_rate is used as the load rate.
                                        Defaults to False.
        callback (function, optional): Callback function. requires arguments
                                     f(particles,material,step).
                                     Defaults to None.
        dim (int, optional): Dimension of simulation.
                             Defaults to 3.

    Returns:
        tuple: particles,material
    """
    load_rates = np.ones(dim) * max_load_rate  # b
    step = 0
    while True:
        stress = particles.stresses[0]

        F = np.array(particles.F[0])
        strain = 0.5 * (F.T + F) - np.identity(dim)

        all_done = np.array([False] * dim)

        for i in range(dim):
            if control[i] == "strain":
                if constant_load:
                    load_rates[i] = max_load_rate

                    if strain[i, i] > targets[i]:
                        all_done[i] = True
                        continue

                error = abs((strain[i, i] - targets[i]) / targets[i])
                if np.abs(error) < tol:
                    all_done[i] = True
                    continue

                est_rate = (
                    (1.0 - strain[i, i] / targets[i])
                    * max_load_rate
                    * np.sign(targets[i])
                    * acceleration
                )

                load_rates[i] = est_rate

            elif control[i] == "stress":
                error = abs((stress[i, i] - targets[i]) / targets[i])
                if abs(error) < tol:
                    all_done[i] = True
                    continue

                est_rate = (
                    (1.0 - stress[i, i] / targets[i])
                    * max_load_rate
                    * np.sign(targets[i])
                    * acceleration
                )

                load_rates[i] = est_rate

        if all_done.all():
            break

        deps = np.identity(dim) * np.diag(load_rates)
        particles.velocity_gradient = [deps]

        particles.F = [
            (np.identity(dim) + np.array(particles.velocity_gradient[0]))
            @ np.array(particles.F[0])
        ]
        particles, _ = material.stress_update(particles, 0)
        step += 1
        if callback is not None:
            callback(particles, material, step)

    return particles, material


def strain_controlled(
    particles,
    material,
    loading_rate,
    total_steps,
    dt,
    callback=None,
    dim=3,
):
    """Strain controlled loading:



    Args:
        particles (ParticlesContainer): PyroclastMPM ParticlesContainer
        material (Material): PyroclastMPM ParticlesContainer
        loading_rate (np.array): strain increment
        total_steps (int): number of steps to load
        callback (function, optional): Callback function. requires arguments
                                     f(particles,material,step).
                                     Defaults to None.
        dim (int, optional): Dimension of simulation.
                             Defaults to 3.

    Returns:
        tuple: particles,material
    """
    velgrad = np.array(loading_rate).astype(np.float32)

    particles.velocity_gradient = [velgrad]

    for step in range(total_steps):
        particles.F = [
            (np.identity(3) + np.array(particles.velocity_gradient[0]) * dt)
            @ np.array(particles.F[0])
        ]
        particles, _ = material.stress_update(particles, 0)

        if callback is not None:
            callback(particles, material, step)

    return particles, material
