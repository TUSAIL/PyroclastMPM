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
 * @file usl.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Update Stress Last (USL) solver
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#include "pyroclastmpm/solver/usl/usl.h"

#include "usl_inline.h"

namespace pyroclastmpm {
///@brief Construct a new USL object
///@param _particles ParticlesContainer class
///@param _nodes NodesContainer class
///@param _materials A list of Materials
///@param _boundaryconditions a list of boundary conditions
///@param _alpha Flip/PIC mixture
USL::USL(const ParticlesContainer &_particles, const NodesContainer &_nodes,
         const cpu_array<MaterialType> &_materials,
         const cpu_array<BoundaryConditionType> &_boundaryconditions,
         Real _alpha)
    : Solver(_particles, _nodes, _materials, _boundaryconditions),
      alpha(_alpha) {}

/// @brief Reset the temporary arrays for the USL solver
void USL::reset() {
  nodes.reset();
  particles.reset();
  particles.spatial.reset();
}

/// @brief Main loop of the USL solver
void USL::solve() {
  reset();

  particles.spawn_particles();

  particles.partition();

  calculate_shape_function(nodes, particles);

  for (auto bc : boundaryconditions) {
    std::visit([this](auto &arg) { arg.apply_on_particles(particles); }, bc);
  }

  P2G();

  for (auto bc : boundaryconditions) {
    std::visit([this](auto &arg) { arg.apply_on_nodes_f_ext(nodes); }, bc);
  }

  nodes.integrate();

  for (auto bc : boundaryconditions) {
    std::visit(
        [this](auto &arg) { arg.apply_on_nodes_moments(nodes, particles); },
        bc);
  }

  G2P();

  stress_update();

  for (auto bc : boundaryconditions) {
    std::visit([this](auto &arg) { arg.apply_on_particles(particles); }, bc);
  }
}

/// @brief Particle to Grid (P2G) operation for USL (velocities gather)
void USL::P2G() {

#ifdef CUDA_ENABLED
  KERNELS_USL_P2G<<<nodes.launch_config.tpb, nodes.launch_config.bpg>>>(
      thrust::raw_pointer_cast(nodes.moments_gpu.data()),
      thrust::raw_pointer_cast(nodes.forces_internal_gpu.data()),
      thrust::raw_pointer_cast(nodes.masses_gpu.data()),
      thrust::raw_pointer_cast(nodes.node_ids_gpu.data()),
      thrust::raw_pointer_cast(particles.stresses_gpu.data()),
      thrust::raw_pointer_cast(particles.forces_external_gpu.data()),
      thrust::raw_pointer_cast(particles.velocities_gpu.data()),
      thrust::raw_pointer_cast(particles.dpsi_gpu.data()),
      thrust::raw_pointer_cast(particles.psi_gpu.data()),
      thrust::raw_pointer_cast(particles.masses_gpu.data()),
      thrust::raw_pointer_cast(particles.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles.spatial.cell_start_gpu.data()),
      thrust::raw_pointer_cast(particles.spatial.cell_end_gpu.data()),
      thrust::raw_pointer_cast(particles.spatial.sorted_index_gpu.data()),
      thrust::raw_pointer_cast(particles.is_rigid_gpu.data()),
      thrust::raw_pointer_cast(particles.is_active_gpu.data()), nodes.grid);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int index = 0; index < nodes.grid.num_cells_total; index++) {

    usl_p2g_kernel(
        nodes.moments_gpu.data(), nodes.forces_internal_gpu.data(),
        nodes.masses_gpu.data(), nodes.node_ids_gpu.data(),
        particles.stresses_gpu.data(), particles.forces_external_gpu.data(),
        particles.velocities_gpu.data(), particles.dpsi_gpu.data(),
        particles.psi_gpu.data(), particles.masses_gpu.data(),
        particles.volumes_gpu.data(), particles.spatial.cell_start_gpu.data(),
        particles.spatial.cell_end_gpu.data(),
        particles.spatial.sorted_index_gpu.data(),
        particles.is_rigid_gpu.data(), particles.is_active_gpu.data(),
        nodes.grid, index);
  }
#endif
}

/// @brief Grid to Particle (G2P) operation for USL (velocities scatter)
void USL::G2P() {
#ifdef CUDA_ENABLED
  KERNEL_USL_G2P<<<particles.launch_config.tpb, particles.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles.F_gpu.data()),
      thrust::raw_pointer_cast(particles.velocities_gpu.data()),
      thrust::raw_pointer_cast(particles.positions_gpu.data()),
      thrust::raw_pointer_cast(particles.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles.dpsi_gpu.data()),
      thrust::raw_pointer_cast(particles.spatial.bins_gpu.data()),
      thrust::raw_pointer_cast(particles.volumes_original_gpu.data()),
      thrust::raw_pointer_cast(particles.psi_gpu.data()),
      thrust::raw_pointer_cast(particles.is_rigid_gpu.data()),
      thrust::raw_pointer_cast(particles.is_active_gpu.data()),
      thrust::raw_pointer_cast(nodes.moments_gpu.data()),
      thrust::raw_pointer_cast(nodes.moments_nt_gpu.data()),
      thrust::raw_pointer_cast(nodes.masses_gpu.data()),
      nodes.grid.particles.num_particles, alpha);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int index = 0; index < particles.num_particles; index++) {
    usl_g2p_kernel(particles.velocity_gradient_gpu.data(),
                   particles.F_gpu.data(), particles.velocities_gpu.data(),
                   particles.positions_gpu.data(), particles.volumes_gpu.data(),
                   particles.dpsi_gpu.data(), particles.spatial.bins_gpu.data(),
                   particles.volumes_original_gpu.data(),
                   particles.psi_gpu.data(), particles.is_rigid_gpu.data(),
                   particles.is_active_gpu.data(), nodes.moments_gpu.data(),
                   nodes.moments_nt_gpu.data(), nodes.masses_gpu.data(),
                   nodes.grid, alpha, index);
  }

#endif
};
} // namespace pyroclastmpm