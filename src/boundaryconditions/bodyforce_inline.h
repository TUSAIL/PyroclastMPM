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

  /**
   * @brief Apply a moment on the background grid
   * @details isFixed means it constraints the nodes
   * @param nodes_moments_nt_gpu forward node moments (see USL)
   * @param nodes_moments_gpu node moments
   * @param values_gpu values of the body moment to be applied
   * @param mask_gpu mask on which nodes to apply the body moment on
   * @param isFixed option to say if the nodes are added or constrained
   * @param node_mem_index index of the node
   */
  __device__ __host__ inline void
  apply_bodymoments(Vectorr *nodes_moments_nt_gpu, Vectorr *nodes_moments_gpu,
                    const Vectorr *values_gpu, const bool *mask_gpu,
                    const bool isFixed, const int node_mem_index)
  {

    if (mask_gpu[node_mem_index])
    {
      if (isFixed)
      {
        // nodes_moments_gpu[node_mem_index] = values_gpu[node_mem_index];
        nodes_moments_nt_gpu[node_mem_index] = values_gpu[node_mem_index];
      }
      else
      {
        nodes_moments_gpu[node_mem_index] += values_gpu[node_mem_index];
      }
    }
  }

#ifdef CUDA_ENABLED
  __global__ void KERNEL_APPLY_BODYMOMENT(Vectorr *nodes_moments_nt_gpu,
                                          Vectorr *nodes_moments_gpu,
                                          const Vectorr *values_gpu,
                                          const bool *mask_gpu,
                                          const bool isFixed,
                                          const int num_nodes_total)
  {

    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    apply_bodymoments(nodes_moments_nt_gpu, nodes_moments_gpu, values_gpu,
                      mask_gpu, isFixed, node_mem_index);
  }

#endif

} // namespace pyroclastmpm