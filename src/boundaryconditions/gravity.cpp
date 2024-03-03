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

/**
 * @file gravity.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Gravity is applied on the background grid through external forces on
 * the nodes
 *
 * @version 0.1
 * @date 2023-06-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "pyroclastmpm/boundaryconditions/gravity.h"

#include "gravity_inline.h"

namespace pyroclastmpm
{

  extern const int global_step_cpu;

  /// @brief Construct a new Gravity object
  /// @param _gravity gravity vector
  /// @param _is_ramp flag whether gravity is ramping linearly or not
  /// @param _ramp_step the amount of steps to ramp gravity to full value
  /// @param _gravity_end gravity value at end of ramp
  Gravity::Gravity(Vectorr _gravity, bool _is_ramp, int _ramp_step,
                   Vectorr _gravity_end)
      : gravity(_gravity), is_ramp(_is_ramp), ramp_step(_ramp_step),
        gravity_end(_gravity_end)
  {

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    spdlog::info("[Gravity] gravity: {}; is_ramp {}; ramp_step {}; gravity_end: {} ", gravity.format(CleanFmt), is_ramp, ramp_step, gravity_end.format(CleanFmt));
  }

  /// @brief Apply graivity to external node forces
  /// @param nodes_ref reference to NodesContainer
  void Gravity::apply_on_nodes_moments(NodesContainer &nodes_ref, ParticlesContainer &particles_ref)
  {

    const Real ramp_factor =
        ((Real)global_step_cpu) / static_cast<Real>(ramp_step);
    if (is_ramp && (global_step_cpu < ramp_step))
    {
      gravity = gravity_end * ramp_factor;
    }
#ifdef CUDA_ENABLED
    KERNEL_APPLY_GRAVITY<<<nodes_ref.launch_config.tpb,
                           nodes_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
        thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()), gravity,
        nodes_ref.grid.num_cells_total);
    gpuErrchk(cudaDeviceSynchronize());
#else
    for (int nid = 0; nid < nodes_ref.grid.num_cells_total; nid++)
    {
      apply_gravity(nodes_ref.moments_nt_gpu.data(),
                    nodes_ref.masses_gpu.data(), gravity, nid);
    }
#endif
  };

} // namespace pyroclastmpm