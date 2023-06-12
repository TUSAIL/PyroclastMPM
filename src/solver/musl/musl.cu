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

#include "pyroclastmpm/solver/musl/musl.cuh"

namespace pyroclastmpm
{

  MUSL::MUSL(
      ParticlesContainer _particles,
      NodesContainer _nodes,
      cpu_array<MaterialType> _materials,
      cpu_array<BoundaryConditionType> _boundaryconditions,
      Real _alpha) : USL(_particles, _nodes, _materials, _boundaryconditions, _alpha)
  {
  }

  void MUSL::solve()
  {
    reset(); // inherited from solver class

    particles.partition();

    calculate_shape_function(); // inherited form solver class

   for (int bc_id = 0; bc_id < boundaryconditions.size(); bc_id++)
    {
      std::visit([&](auto &arg)
                 { arg.apply_on_particles(particles); },
                 boundaryconditions[bc_id]);
    }
    
    P2G(); // inherited from usl class

    for (int bc_id = 0; bc_id < boundaryconditions.size(); bc_id++)
    {
      std::visit([&](auto &arg)
                 { arg.apply_on_nodes_f_ext(nodes); },
                 boundaryconditions[bc_id]);
    }

    nodes.integrate();

    for (int bc_id = 0; bc_id < boundaryconditions.size(); bc_id++)
    {
      std::visit([&](auto &arg)
                 { arg.apply_on_nodes_moments(nodes, particles); },
                 boundaryconditions[bc_id]);
    }

    G2P_double_mapping();

    P2G_double_mapping();

    for (int bc_id = 0; bc_id < boundaryconditions.size(); bc_id++)
    {
      std::visit([&](auto &arg)
                 { arg.apply_on_nodes_moments(nodes, particles); },
                 boundaryconditions[bc_id]);
    }

    G2P();

    stress_update(); // inherited from solver class

    for (int bc_id = 0; bc_id < boundaryconditions.size(); bc_id++)
    {
      std::visit([&](auto &arg)
                 { arg.apply_on_particles(particles); },
                 boundaryconditions[bc_id]);
    }
  }

  void MUSL::P2G_double_mapping()
  {

    KERNEL_MUSL_P2G_DOUBLE_MAPPING<<<nodes.launch_config.tpb,
                                     nodes.launch_config.bpg>>>(
        thrust::raw_pointer_cast(nodes.moments_nt_gpu.data()),
        thrust::raw_pointer_cast(nodes.masses_gpu.data()),
        thrust::raw_pointer_cast(nodes.node_ids_gpu.data()),
        thrust::raw_pointer_cast(particles.velocities_gpu.data()),
        thrust::raw_pointer_cast(particles.dpsi_gpu.data()),
        thrust::raw_pointer_cast(particles.psi_gpu.data()),
        thrust::raw_pointer_cast(particles.masses_gpu.data()),
        thrust::raw_pointer_cast(particles.spatial.cell_start_gpu.data()),
        thrust::raw_pointer_cast(particles.spatial.cell_end_gpu.data()),
        thrust::raw_pointer_cast(particles.spatial.sorted_index_gpu.data()),
        nodes.num_nodes, nodes.num_nodes_total);

    gpuErrchk(cudaDeviceSynchronize());
  }

  void MUSL::G2P_double_mapping()
  {

    KERNEL_MUSL_G2P_DOUBLE_MAPPING<<<particles.launch_config.tpb,
                                     particles.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles.velocities_gpu.data()),
        thrust::raw_pointer_cast(particles.positions_gpu.data()),
        thrust::raw_pointer_cast(particles.dpsi_gpu.data()),
        // thrust::raw_pointer_cast(nodes.node_ids_gpu.data()),
        thrust::raw_pointer_cast(particles.spatial.bins_gpu.data()),
        thrust::raw_pointer_cast(particles.psi_gpu.data()),
        thrust::raw_pointer_cast(nodes.moments_gpu.data()),
        thrust::raw_pointer_cast(nodes.moments_nt_gpu.data()),
        thrust::raw_pointer_cast(nodes.masses_gpu.data()),
        particles.spatial.num_cells,
        particles.num_particles, alpha);
    gpuErrchk(cudaDeviceSynchronize());
  };

  void MUSL::G2P()
  {
    KERNEL_MUSL_G2P<<<particles.launch_config.tpb,
                      particles.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles.velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(particles.F_gpu.data()),
        thrust::raw_pointer_cast(particles.volumes_gpu.data()),
        thrust::raw_pointer_cast(nodes.moments_nt_gpu.data()),
        thrust::raw_pointer_cast(particles.dpsi_gpu.data()),
        thrust::raw_pointer_cast(particles.spatial.bins_gpu.data()),
        thrust::raw_pointer_cast(particles.volumes_original_gpu.data()),
        thrust::raw_pointer_cast(particles.psi_gpu.data()),
        thrust::raw_pointer_cast(particles.masses_gpu.data()),
        thrust::raw_pointer_cast(nodes.masses_gpu.data()),
        particles.spatial.num_cells,
        particles.num_particles);

    gpuErrchk(cudaDeviceSynchronize());
  };

} // namespace pyroclastmpm