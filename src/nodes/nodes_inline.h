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
#pragma once
/**
 * @file nodes_inline.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Contains main kernels for nodal integration
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */
#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern const Real dt_cpu;
#endif

/** @brief Integrate nodal arrays to get moments
 *
 * @param nodes_moments_nt_gpu Forwards nodal moments
 * @param nodes_forces_total_gpu Total nodal forces
 * @param nodes_forces_external_gpu External nodal forces
 * @param nodes_forces_internal_gpu Internal nodal forces
 * @param nodes_moments_gpu Nodal moments
 * @param nodes_masses_gpu Nodal masses
 * @param node_mem_index Index of node in memory
 * @return __device__
 */
__device__ __host__ inline void
integrate_nodes(Vectorr *nodes_moments_nt_gpu, Vectorr *nodes_forces_total_gpu,
                const Vectorr *nodes_forces_external_gpu,
                const Vectorr *nodes_forces_internal_gpu,
                const Vectorr *nodes_moments_gpu, const Real *nodes_masses_gpu,
                const Real small_mass_cutoff, const int node_mem_index) {

  if (nodes_masses_gpu[node_mem_index] <= small_mass_cutoff) {
    return;
  }
  const Vectorr ftotal = nodes_forces_internal_gpu[node_mem_index] +
                         nodes_forces_external_gpu[node_mem_index];

  nodes_forces_total_gpu[node_mem_index] = ftotal;

  // moments_nt - updated moments
  // moments - old moments
#ifdef CUDA_ENABLED
  nodes_moments_nt_gpu[node_mem_index] =
      nodes_moments_gpu[node_mem_index] + ftotal * dt_gpu;
#else
  nodes_moments_nt_gpu[node_mem_index] =
      nodes_moments_gpu[node_mem_index] + ftotal * dt_cpu;
#endif
}

#ifdef CUDA_ENABLED
__global__ void
KERNEL_INTEGRATE(Vectorr *nodes_moments_nt_gpu, Vectorr *nodes_forces_total_gpu,
                 const Vectorr *nodes_forces_external_gpu,
                 const Vectorr *nodes_forces_internal_gpu,
                 const Vectorr *nodes_moments_gpu, const Real *nodes_masses_gpu,
                 const Real small_mass_cutoff, const int num_nodes_total) {

  const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (node_mem_index >= num_nodes_total) {
    return;
  }

  integrate_nodes(nodes_moments_nt_gpu, nodes_forces_total_gpu,
                  nodes_forces_external_gpu, nodes_forces_internal_gpu,
                  nodes_moments_gpu, nodes_masses_gpu, small_mass_cutoff,
                  node_mem_index);
}

#endif

} // namespace pyroclastmpm