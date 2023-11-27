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

from .basecontrol import BaseControl


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
    debug=False,
):
    # strain goal of target strain tensor with guess
    # if successful except strain tensor
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
    # print(f"{velgrad=}")
    particles.F = [strain_trail]

    # update yield surface and history variables
    material.do_update_history = do_update_history

    material.is_velgrad_strain_increment = ~is_finite_strain

    particles, _ = material.stress_update(particles, 0)

    curr_stress_11_22 = np.array(particles.stresses[0])[(1, 2), (1, 2)]
    if do_update_history:
        if debug:
            print(
                f"{do_update_history=} {curr_stress_11_22=}, {target_radial_stress_11_22=}"
            )
        return particles, material

    if debug:
        print(
            f"{do_update_history=} {curr_stress_11_22=}, {target_radial_stress_11_22=} {velgrad=}"
        )
    stress_11_22_diff = curr_stress_11_22 - target_radial_stress_11_22
    normalized_stress_11_22_diff = np.linalg.norm(stress_11_22_diff) ** 2
    # print(normalized_stress_11_22_diff)
    return normalized_stress_11_22_diff


class TriaxialControl(BaseControl):
    """

    modes: sequence - takes a strain_target sequence
    mode: "range" - takes a range of strain targets
    _summary_
    """

    def run(self, debug=False):
        if self.mode is None:
            raise ValueError("Mode not set")

        # sigma11 and sigma22 are the same
        target_radial_stress_11_22 = np.ones(2) * self.radial_stress_target

        for step in range(self.num_steps):
            # start from prestrain
            target_strain_tensor = self.prestrain.copy()
            target_strain_tensor[0, 0] += self.axial_strain_target_list[step]

            # exit()
            # initial guess / starting point (eps11,eps22) is previous step
            strain_guess_11_22 = np.array(self.strain_prev)[(1, 2), (1, 2)]

            # if step == 0:
            # self.store_results(step)
            # self.strain_prev = -1*np.ones(2)
            # continue # skip first step
            res = scipy.optimize.minimize(
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
                    debug,
                ),
                tol=self.tolerance,
                # method=self.optimize_method,
                method="Nelder-Mead",
                # bounds=scipy.optimize.Bounds(-1e-13, 1e-3, True),
                options={
                    # "xatol": 1e-6
                    # maxiter =100
                    # "ftol": self.tolerance,
                    # "line_search": None,
                    # "xtol": 1e-6,
                    # "xatol": 1e-12,
                    # "jac_options": {"rdiff": 1e-8},
                    # "disp": True,
                },
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
                self.store_results(step)  # defined in baseclass

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
