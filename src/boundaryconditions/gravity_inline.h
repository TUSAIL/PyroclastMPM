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

#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm
{

#ifdef CUDA_ENABLED
  extern __constant__ Real dt_gpu;
#else
  extern const Real dt_cpu;
#endif

  /**
   * @brief Apply gravity to the nodes of the background grid. *
   * @details The nodes must have a mass mass > 0.000000001
   *
   * @param nodes_forces_external_gpu external forces of the background grid
   * @param nodes_masses_gpu masses of the nodes
   * @param gravity gravity vector
   * @param node_mem_index index of the node
   */
  __device__ __host__ inline void
  apply_gravity(Vectorr *nodes_moments_nt_gpu, const Real *nodes_masses_gpu,
                const Vectorr gravity, const int node_mem_index)
  {

    const Real node_mass = nodes_masses_gpu[node_mem_index];

    if (node_mass <= 0.000000001)
    {
      return;
    }
    // moment = mass * gravity * dt
#ifdef CUDA_ENABLED
    nodes_moments_nt_gpu[node_mem_index] += gravity * node_mass * dt_gpu;
#else
    nodes_moments_nt_gpu[node_mem_index] += gravity * node_mass * dt_cpu;
#endif
  }

#ifdef CUDA_ENABLED
  __global__ void KERNEL_APPLY_GRAVITY(Vectorr *nodes_moments_nt_gpu,
                                       const Real *nodes_masses_gpu,
                                       const Vectorr gravity,
                                       const int num_nodes_total)

  {

    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    apply_gravity(nodes_moments_nt_gpu, nodes_masses_gpu, gravity,
                  node_mem_index);
  }

#endif

} // namespace pyroclastmpm