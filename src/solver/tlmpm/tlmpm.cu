#include "pyroclastmpm/solver/tlmpm/tlmpm.cuh"

namespace pyroclastmpm
{

  extern int global_step_cpu;

  TLMPM::TLMPM(
      ParticlesContainer _particles,
      NodesContainer _nodes,
      cpu_array<MaterialType> _materials,
      cpu_array<BoundaryConditionType> _boundaryconditions,
      Real _alpha) : MUSL(_particles, _nodes, _materials, _boundaryconditions, _alpha)
  {
    particles.partition();
    calculate_shape_function();
    set_default_device<Matrix3r>(_particles.num_particles, {}, stresses_pk1_gpu, Matrix3r::Zero());
    // add array PK stress
  }
  /**
   * @brief Reset the temporary arrays for the TLMPM solver
   *
   */
  void TLMPM::reset()
  {
    nodes.reset();
    particles.reset(false);
  }

  /**
   * @brief Main loop of the TLMPM solver
   *
   */
  void TLMPM::solve()
  {
    reset();

    for (int bc_id = 0; bc_id < boundaryconditions.size(); bc_id++)
    {
      std::visit([&](auto &arg)
                 { arg.apply_on_particles(particles); },
                 boundaryconditions[bc_id]);
    }

    P2G();

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

    CauchyStressToPK1Stress();
  }
  

  void TLMPM::P2G()
  {
    KERNELS_USL_P2G<<<nodes.launch_config.tpb,
                      nodes.launch_config.bpg>>>(
        thrust::raw_pointer_cast(nodes.moments_gpu.data()),
        thrust::raw_pointer_cast(nodes.forces_internal_gpu.data()),
        thrust::raw_pointer_cast(nodes.masses_gpu.data()),
        thrust::raw_pointer_cast(nodes.node_ids_gpu.data()),
        thrust::raw_pointer_cast(stresses_pk1_gpu.data()), // let this be PK stress
        thrust::raw_pointer_cast(particles.forces_external_gpu.data()),
        thrust::raw_pointer_cast(particles.velocities_gpu.data()),
        thrust::raw_pointer_cast(particles.dpsi_gpu.data()),
        thrust::raw_pointer_cast(particles.psi_gpu.data()),
        thrust::raw_pointer_cast(particles.masses_gpu.data()),
        // thrust::raw_pointer_cast(particles.volumes_gpu.data()),
        thrust::raw_pointer_cast(particles.volumes_original_gpu.data()), // use original volume
        thrust::raw_pointer_cast(particles.spatial.cell_start_gpu.data()),
        thrust::raw_pointer_cast(particles.spatial.cell_end_gpu.data()),
        thrust::raw_pointer_cast(particles.spatial.sorted_index_gpu.data()),
        nodes.num_nodes, nodes.inv_node_spacing, nodes.num_nodes_total);

    gpuErrchk(cudaDeviceSynchronize());
  }

    void TLMPM::G2P_double_mapping()
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
        particles.num_particles, alpha,true);
    gpuErrchk(cudaDeviceSynchronize());
  };

  /**
   * @brief Grid to Particle (G2P) operation for TLMPM (velocities scatter)
   *
   */
  void TLMPM::G2P()
  {
    KERNEL_TLMPM_G2P<<<particles.launch_config.tpb,
                       particles.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles.velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(particles.F_gpu.data()),
        thrust::raw_pointer_cast(particles.volumes_gpu.data()),
        thrust::raw_pointer_cast(particles.positions_gpu.data()),
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

  void TLMPM::CauchyStressToPK1Stress()
  {
    KERNEL_TLMPM_CONVERT_STRESS<<<particles.launch_config.tpb,
                                  particles.launch_config.bpg>>>(
        thrust::raw_pointer_cast(stresses_pk1_gpu.data()),
        thrust::raw_pointer_cast(particles.stresses_gpu.data()),
        thrust::raw_pointer_cast(particles.F_gpu.data()),
        particles.num_particles);
    gpuErrchk(cudaDeviceSynchronize());
  }

} // namespace pyroclastmpm