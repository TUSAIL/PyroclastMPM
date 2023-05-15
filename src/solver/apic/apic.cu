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