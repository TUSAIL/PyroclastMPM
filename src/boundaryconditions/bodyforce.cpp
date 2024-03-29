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
 * @file body_force.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief This file contains methods to apply a body force or moments nodes
 *
 * @version 0.1
 * @date 2023-06-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "pyroclastmpm/boundaryconditions/bodyforce.h"
#include "bodyforce_inline.h"

namespace pyroclastmpm
{

  /// @brief Construct a new Body Force object
  /// @param _mode mode of the body force
  /// @param _values values of the body force
  /// @param _mask mask on which nodes to apply the body force
  BodyForce::BodyForce(const std::string_view &_mode,
                       const cpu_array<Vectorr> &_values,
                       const cpu_array<bool> &_mask) noexcept
  {
    if (_mode == "forces")
    {
      mode_id = 0;
    }
    else if (_mode == "moments")
    {
      mode_id = 1;
    }
    else if (_mode == "fixed")
    {
      mode_id = 2;
    }

    set_default_device<Vectorr>(static_cast<int>(_values.size()), _values,
                                values_gpu, Vectorr::Zero());
    set_default_device<bool>(static_cast<int>(_mask.size()), _mask, mask_gpu,
                             false);
  }

  /// @brief Update node values for moments
  /// @param nodes_ref NodeContainer reference
  /// @param particles_ref ParticleContainer reference
  void BodyForce::apply_on_nodes_moments(NodesContainer &nodes_ref,
                                         ParticlesContainer &particles_ref)
  {
    if (!isActive)
    {
      return;
    }

    bool isFixed = (mode_id == 2);

#ifdef CUDA_ENABLED
    KERNEL_APPLY_BODYMOMENT<<<nodes_ref.launch_config.tpb,
                              nodes_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
        thrust::raw_pointer_cast(nodes_ref.moments_gpu.data()),
        thrust::raw_pointer_cast(values_gpu.data()),
        thrust::raw_pointer_cast(mask_gpu.data()), isFixed,
        nodes_ref.grid.num_cells_total);
#else
    for (int nid = 0; nid < nodes_ref.grid.num_cells_total; nid++)
    {

      apply_bodymoments(nodes_ref.moments_nt_gpu.data(),
                        nodes_ref.moments_gpu.data(), values_gpu.data(),
                        mask_gpu.data(), isFixed, nid);
    }
#endif
  };

} // namespace pyroclastmpm