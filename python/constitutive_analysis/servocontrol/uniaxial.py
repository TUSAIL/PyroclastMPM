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


class UniaxialControl(BaseControl):
    """

    modes: sequence - takes a strain_target sequence
    mode: "range" - takes a range of strain targets
    _summary_
    """

    def run(self):
        for step in range(self.num_steps):
            target_strain_tensor = self.prestrain.copy()
            target_strain_tensor[0, 0] += -self.axial_strain_target_list[step]
            # print(target_strain_tensor)
            current_volume = self.particles.volumes[0]

            if self.is_finite_strain:
                dFdt = (target_strain_tensor - self.strain_prev) / self.dt
                velgrad = dFdt @ np.linalg.inv(target_strain_tensor)
            else:
                velgrad = target_strain_tensor - self.strain_prev
                volume = current_volume * (1 + np.trace(velgrad))

            self.particles.volumes = [volume]
            self.particles.velocity_gradient = [velgrad]
            # print(f"{velgrad=}")
            self.particles.F = [target_strain_tensor]

            # update yield surface and history variables
            self.material.do_update_history = True

            self.material.is_velgrad_strain_increment = ~self.is_finite_strain

            self.particles, _ = self.material.stress_update(self.particles, 0)

            self.strain_prev = np.array(self.particles.F[0])

            if step % self.output_step == 0:
                self.store_results(step)  # defined in baseclass

    def set_mode_sequence(self, axial_strain_target_list, total_time):
        """
        Set the mode to "sequence" and set the sequence of strain targets.
        """
        self.mode = "sequence"
        self.axial_strain_target_list = axial_strain_target_list
        self.total_time = total_time
        self.num_steps = len(axial_strain_target_list)
        self.timestep = total_time / self.num_steps
        pm.set_global_timestep(self.timestep)

    def set_mode_range(
        self,
        axial_strain_target,
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

        self.total_time = total_time
        self.timestep = timestep
        pm.set_global_timestep(self.timestep)
