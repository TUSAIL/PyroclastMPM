// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "pyroclastmpm/boundaryconditions/gravity.h"

namespace pyroclastmpm {

extern int global_step_cpu;

#include "gravity_inline.h"

/**
 * @brief Gravity boundary condition, either constant or linear ramping
 *
 * @param _gravity initial gravity vector
 * @param _is_ramp whether the gravity is linear ramping or not
 * @param _ramp_step time when full gravity is reached
 * @param _gravity_end gravity value at end of ramp
 */
Gravity::Gravity(Vectorr _gravity, bool _is_ramp, int _ramp_step,
                 Vectorr _gravity_end) {
  gravity = _gravity;
  is_ramp = _is_ramp;
  ramp_step = _ramp_step;
  gravity_end = _gravity_end;
}

void Gravity::apply_on_nodes_f_ext(NodesContainer &nodes_ref) {

  const Real ramp_factor = ((Real)global_step_cpu) / ramp_step;
  if (is_ramp) {

    if (global_step_cpu < ramp_step) {
      gravity = gravity_end * ramp_factor;
    }
  }
#ifdef CUDA_ENABLED
  KERNEL_APPLY_GRAVITY<<<nodes_ref.launch_config.tpb,
                         nodes_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(nodes_ref.forces_external_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()), gravity,
      nodes_ref.num_nodes_total);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int nid = 0; nid < nodes_ref.num_nodes_total; nid++) {
    apply_gravity(nodes_ref.forces_external_gpu.data(),
                  nodes_ref.masses_gpu.data(), gravity, nid);
  }
#endif
};

} // namespace pyroclastmpm