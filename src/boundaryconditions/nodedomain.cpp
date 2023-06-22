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

#include "pyroclastmpm/boundaryconditions/nodedomain.h"
#include "nodedomain_inline.h"

namespace pyroclastmpm {
/// @brief Construct a new object
/// @param face0_mode roller or fixed modes for cube face x0,y0,z0
/// @param face1_mode roller or fixed modes  for cube face x1,y1,z1
NodeDomain::NodeDomain(Vectori _face0_mode, Vectori _face1_mode)
    : face1_mode(_face1_mode), face0_mode(_face0_mode) {}

/// @brief Apply to node moments (walls)
/// @param nodes_ref reference to NodesContainer
void NodeDomain::apply_on_nodes_moments(NodesContainer &nodes_ref,
                                        ParticlesContainer &particles_ref) {

#ifdef CUDA_ENABLED
  KERNEL_APPLY_NODEDOMAIN<<<nodes_ref.launch_config.tpb,
                            nodes_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.moments_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
      nodes_ref.grid.num_cells, axis0_mode, axis1_mode,
      nodes_ref.grid.num_cells_total);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int nid = 0; nid < nodes_ref.grid.num_cells_total; nid++) {

    apply_nodedomain(nodes_ref.moments_nt_gpu.data(),
                     nodes_ref.moments_gpu.data(), nodes_ref.masses_gpu.data(),
                     nodes_ref.node_ids_gpu.data(), nodes_ref.grid.num_cells,
                     face0_mode, face1_mode, nid);
  }
#endif
};

} // namespace pyroclastmpm