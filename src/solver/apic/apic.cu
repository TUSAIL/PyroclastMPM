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

#include "pyroclastmpm/solver/apic/apic.cuh"

namespace pyroclastmpm {

APIC::APIC(ParticlesContainer _particles,
           NodesContainer _nodes,
           thrust::host_vector<MaterialType> _materials,
           thrust::host_vector<BoundaryConditionType> _boundaryconditions)
    : Solver(_particles, _nodes, _materials, _boundaryconditions)

{
  // APIC;
  printf("init APIC \n");

  // set_default_device(_particles_ptr->num_particles, {}, Bp_gpu);

  const Matrix3r Wp = (1 / 3.) * _nodes.node_spacing * _nodes.node_spacing *
                      Matrix3r::Identity();  // only works for cubic splines

  Wp_inverse = Wp.inverse();

  gpuErrchk(cudaDeviceSynchronize());
}
void APIC::reset() {
  nodes.reset();
  particles.reset();
  particles.spatial.reset();
}

void APIC::solve() {
  reset();

  particles.partition();

  calculate_shape_function();

  P2G();

  // for (auto& bc : boundaryconditions) {
  //   bc->apply_on_nodes_f_ext(nodes);
  // }

  nodes.integrate();

  // for (auto& bc : boundaryconditions) {
  //   bc->apply_on_nodes_moments(nodes, particles);
  // }

  G2P();

  // particles_ptr->calculate_stresses();

  // for (auto& bc : boundaryconditions) {
  //   bc->apply_on_particles(particles);
  // }
}

void APIC::P2G() {
  KERNELS_USL_P2G_APIC<<<nodes.launch_grid_config_p2g,
                         nodes.launch_block_config_p2g>>>(
      thrust::raw_pointer_cast(nodes.moments_gpu.data()),
      thrust::raw_pointer_cast(nodes.forces_internal_gpu.data()),
      thrust::raw_pointer_cast(nodes.masses_gpu.data()),
      thrust::raw_pointer_cast(nodes.node_ids_3d_gpu.data()),
      thrust::raw_pointer_cast(particles.stresses_gpu.data()),
      thrust::raw_pointer_cast(particles.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles.velocities_gpu.data()),
      thrust::raw_pointer_cast(particles.positions_gpu.data()),
      thrust::raw_pointer_cast(particles.dpsi_gpu.data()),
      thrust::raw_pointer_cast(particles.psi_gpu.data()),
      thrust::raw_pointer_cast(particles.masses_gpu.data()),
      thrust::raw_pointer_cast(particles.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles.spatial.cell_start_gpu.data()),
      thrust::raw_pointer_cast(particles.spatial.cell_end_gpu.data()),
      thrust::raw_pointer_cast(particles.spatial.sorted_index_gpu.data()),
      nodes.num_nodes, nodes.inv_node_spacing, nodes.num_nodes_total);

  gpuErrchk(cudaDeviceSynchronize());
}

void APIC::G2P() {
  printf("APIC G2P: APIC NOT WORKING! \n");
  exit(0);
  // KERNEL_USL_G2P_APIC<<<particles_ptr->launch_tpb,
  // particles_ptr->launch_nb>>>(
  //     thrust::raw_pointer_cast(particles_ptr->velocity_gradient_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->F_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->strains_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->strain_increments_gpu.data()),
  //     // thrust::raw_pointer_cast(particles_ptr->strain_rates_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->velocities_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->positions_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->volumes_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->densities_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->dpsi_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->spatial_ptr->bins_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->volumes_original_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->densities_original_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->psi_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ptr->masses_gpu.data()),
  //     thrust::raw_pointer_cast(nodes_ptr->moments_gpu.data()),
  //     thrust::raw_pointer_cast(nodes_ptr->moments_nt_gpu.data()),
  //     thrust::raw_pointer_cast(nodes_ptr->masses_gpu.data()),
  //     Wp_inverse, nodes_ptr->inv_node_spacing,
  //     particles_ptr->spatial_ptr->num_cells, particles_ptr->num_particles);
  gpuErrchk(cudaDeviceSynchronize());
};

}  // namespace pyroclastmpm