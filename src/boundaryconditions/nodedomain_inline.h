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

namespace pyroclastmpm {
/**
 * @brief Apply wall boundary conditions to nodes
 * @param nodes_moments_nt_gpu Forward node moments (see USL)
 * @param nodes_moments_gpu node moments
 * @param nodes_masses_gpu node masses
 * @param nodes_bins_gpu node bins
 * @param num_nodes number of nodes
 * @param face0_mode face 0 mode
 * @param face1_mode face 1 mode
 * @param node_mem_index node memory index
 */
__device__ __host__ inline void
apply_nodedomain(Vectorr *nodes_moments_nt_gpu, Vectorr *nodes_moments_gpu,
                 const Real *nodes_masses_gpu, const Vectori *nodes_bins_gpu,
                 const Vectori num_nodes, const Vectori face0_mode,
                 const Vectori face1_mode, const int node_mem_index) {

#ifndef CUDA_ENABLED
  // to call std::max on CPU and avoid error occurring without max:
  // `there are no arguments to 'max' that depend on a template parameter...`
  using namespace std;
#endif

  if (const Real node_mass = nodes_masses_gpu[node_mem_index];
      node_mass <= 0.00001) {
    return;
  }
  const Vectori node_bin = nodes_bins_gpu[node_mem_index];

  for (int i = 0; i < DIM; i++) {
    if (node_bin[i] < 1) {

      if (face0_mode(i) == 0) {
        nodes_moments_gpu[node_mem_index] = Vectorr::Zero();
        nodes_moments_nt_gpu[node_mem_index] = Vectorr::Zero();
      } else if (face0_mode(i) == 1) {
        // stick
        nodes_moments_gpu[node_mem_index][i] =
            (Real)max(0., nodes_moments_gpu[node_mem_index]
                              .cast<double>()[i]); // cast to double to avoid
                                                   // error with std::max
        nodes_moments_nt_gpu[node_mem_index][i] = (Real)max(
            0., nodes_moments_nt_gpu[node_mem_index].cast<double>()[i]);
      }
    } else if (node_bin[i] >= num_nodes(i) - 1) {
      if (face1_mode(i) == 0) {
        nodes_moments_gpu[node_mem_index] = Vectorr::Zero();
        nodes_moments_nt_gpu[node_mem_index] = Vectorr::Zero();
      } else if (face1_mode(i) == 1) {
        // slip
        nodes_moments_gpu[node_mem_index][i] =
            (Real)min(0., nodes_moments_gpu[node_mem_index].cast<double>()[i]);
        nodes_moments_nt_gpu[node_mem_index][i] = (Real)min(
            0., nodes_moments_nt_gpu[node_mem_index].cast<double>()[i]);
      }
    }
  }
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_APPLY_NODEDOMAIN(
    Vectorr *nodes_moments_nt_gpu, Vectorr *nodes_moments_gpu,
    const Real *nodes_masses_gpu, const Vectori *nodes_bins_gpu,
    const Vectori num_nodes, const Vectori face0_mode, const Vectori face1_mode,
    const int num_nodes_total)

{
  const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (node_mem_index >= num_nodes_total) {
    return;
  }

  apply_nodedomain(nodes_moments_nt_gpu, nodes_moments_gpu, nodes_masses_gpu,
                   nodes_bins_gpu, num_nodes, face0_mode, face1_mode,
                   node_mem_index);
}

#endif

} // namespace pyroclastmpm