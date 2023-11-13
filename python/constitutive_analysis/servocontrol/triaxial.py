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
import pyroclastmpm.MPM3D as pm
import scipy


def triax_loading_path(
    strain_guess_11_22,
    target_radial_stress_11_22,
    target_strain_tensor,
    strain_prev,
    dt,
    particles,
    material,
    do_update_history=False,
    is_finite_strain=False,
):
    # strain goal of target strain tensor with guess
    # if successfull except strain tensor
    # else make new guess
    strain_trail = target_strain_tensor.copy()
    strain_trail[(1, 2), (1, 2)] = strain_guess_11_22

    current_volume = particles.volumes[0]
    # use the strain increment instead of velocty gradient
    if is_finite_strain:
        dFdt = (strain_trail - strain_prev) / dt
        velgrad = dFdt @ np.linalg.inv(strain_trail)
    else:
        velgrad = strain_trail - strain_prev
        volume = current_volume * (1 + np.trace(velgrad))

    particles.volumes = [volume]
    particles.velocity_gradient = [velgrad]

    particles.F = [strain_trail]

    # update yield surface and history variables
    material.do_update_history = do_update_history

    material.is_velgrad_strain_increment = ~is_finite_strain

    particles, _ = material.stress_update(particles, 0)

    if do_update_history:
        return particles, material

    curr_stress_11_22 = np.array(particles.stresses[0])[(1, 2), (1, 2)]

    return (curr_stress_11_22 - target_radial_stress_11_22) / abs(
        target_radial_stress_11_22[0]
    )


class TriaxialControl:
    """

    modes: sequence - takes a strain_target sequence
    mode: "range" - takes a range of strain targets
    _summary_
    """

    def run(self):
        if self.mode is None:
            raise ValueError("Mode not set")

        # sigma11 and sigma22 are the same
        target_radial_stress_11_22 = np.ones(2) * self.radial_stress_target
        # self.store_results(-1)
        for step in range(self.num_steps):
            # start from prestrain
            target_strain_tensor = self.prestrain.copy()
            target_strain_tensor[0, 0] += self.axial_strain_target_list[step]

            # exit()
            # initial guess / starting point (eps11,eps22) is previous step
            strain_guess_11_22 = np.array(self.strain_prev)[(1, 2), (1, 2)]

            res = scipy.optimize.root(
                triax_loading_path,
                strain_guess_11_22,
                args=(
                    target_radial_stress_11_22,
                    target_strain_tensor,
                    self.strain_prev,
                    self.timestep,
                    self.particles,
                    self.material,
                    False,
                    self.is_finite_strain,
                ),
                method=self.optimize_method,
                # tol=self.tolerance,
                options={"fatol": self.tolerance, "line_search": "wolfe"},
            )

            # return

            self.particles, self.material = triax_loading_path(
                res.x,
                target_radial_stress_11_22,
                target_strain_tensor,
                self.strain_prev,
                self.timestep,
                self.particles,
                self.material,
                do_update_history=True,
                is_finite_strain=self.is_finite_strain,
            )
            self.strain_prev = np.array(self.particles.F[0])
            self.strain_prev[(1, 2), (1, 2)] = res.x

            if step % self.output_step == 0:
                self.store_results(step)

        # relative stress

    def store_results(self, step):
        """
        Store the results in a list.
        """
        if step == -1:
            stress = np.array(self.prestress) * self.sign_flip
            strain = np.array(self.prestrain) * self.sign_flip
        else:
            stress = np.array(self.particles.stresses[0]) * self.sign_flip
            strain = np.array(self.particles.F[0]) * self.sign_flip

        # print(stress, strain)
        self.stress_list.append(stress)
        self.strain_list.append(strain)
        self.step_list.append(step)

        if self.verbose:
            print(f"storing array step {step}")

    def set_mode_sequence(
        self, axial_strain_target_list, radial_stress_target, total_time
    ):
        """
        Set the mode to "sequence" and set the sequence of strain targets.
        """
        self.mode = "sequence"
        self.axial_strain_target_list = axial_strain_target_list
        self.radial_stress_target = radial_stress_target
        self.total_time = total_time
        self.num_steps = len(axial_strain_target_list)
        self.timestep = total_time / self.num_steps
        pm.set_global_timestep(self.timestep)

    def set_mode_range(
        self,
        axial_strain_target,
        radial_stress_target,
        total_time,
        timestep,
    ):
        """
        Set the mode to "range" and set the end points of the range.
        """
        self.mode = "range"
        self.num_steps = int(total_time / timestep)
        self.axial_strain_target_list = np.linspace(
            0, axial_strain_target, self.num_steps
        )

        self.radial_stress_target = radial_stress_target
        self.total_time = total_time
        self.timestep = timestep
        pm.set_global_timestep(self.timestep)

    def set_sign_flip(self, sign_flip):
        self.sign_flip = sign_flip

    def post_process(self):
        self.stress_list = np.array(self.stress_list)
        self.strain_list = np.array(self.strain_list)

        self.pressure_list = -(1 / 3.0) * (
            self.stress_list[:, 0, 0]
            + self.stress_list[:, 1, 1]
            + self.stress_list[:, 2, 2]
        )

        self.dev_stress_list = (
            self.stress_list
            + np.identity(3) * self.pressure_list[:, None, None]
        )

        self.q_vm = np.array(
            list(
                map(
                    lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
                    self.dev_stress_list,
                )
            )
        )

    def __init__(
        self,
        particles,
        material,
        tolerance=1e-7,
        optimize_method="krylov",
        is_finite_strain=False,
        is_staged=False,
        output_step=100,
        verbose=False,
    ):
        self.particles = particles
        self.material = material
        self.tolerance = tolerance
        self.optimize_method = optimize_method
        self.is_finite_strain = is_finite_strain
        self.is_staged = is_staged
        self.mode = None
        self.output_step = output_step
        self.verbose = verbose

        self.sign_flip = 1.0  # or -1.0

        self.prestress = np.array(self.particles.stresses[0])

        self.stress_list = []
        self.strain_list = []
        self.step_list = []

        if is_staged:
            self.prestrain = np.array(self.particles.F[0])
        else:
            if is_finite_strain:
                self.prestrain = np.identity(3)
            else:
                self.prestrain = np.zeros((3, 3))

        self.strain_prev = self.prestrain

    # def plot_strain_grid(self, file=None):
    #     import matplotlib.pyplot as plt

    #     _, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False)

    #     strain_list_np = np.array(self.strain_list)

    #     for i in range(3):
    #         for j in range(3):
    #             axes[i, j].plot(self.step_list, strain_list_np[:, i, j])

    #     if file is not None:
    #         plt.savefig(file)

    # def plot_q_p(self, file=None):
    #     import matplotlib.pyplot as plt

    #     stress_list_np = np.array(self.stress_list)

    #     pressure = -(1 / 3.0) * (
    #         stress_list_np[:, 0, 0]
    #         + stress_list_np[:, 1, 1]
    #         + stress_list_np[:, 2, 2]
    #     )

    #     dev_stress = stress_list_np + np.identity(3) * pressure[:, None, None]

    #     q_vomMises = np.array(
    #         list(
    #             map(
    #                 lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
    #                 dev_stress,
    #             )
    #         )
    #     )
    #     fig, ax = plt.subplots()

    #     ax.plot(pressure, q_vomMises)
    #     if file is not None:
    #         plt.savefig(file)

    # def plot_q_eps11(self, file=None):
    #     import matplotlib.pyplot as plt

    #     stress_list_np = np.array(self.stress_list)
