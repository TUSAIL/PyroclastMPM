#include "pyroclastmpm/solver/usl/usl.cuh"

namespace pyroclastmpm
{

#ifdef CUDA_ENABLED
  extern __constant__ Real dt_gpu;
  extern __constant__ int num_surround_nodes_gpu;
  extern __constant__ int forward_window_gpu[64][3];
  extern __constant__ int backward_window_gpu[64][3];
#else
  extern Real dt_cpu;
  extern int num_surround_nodes_cpu;
  extern int forward_window_cpu[64][3];
  extern int backward_window_cpu[64][3];
#endif

  // include private header with kernels here to inline them
  #include "usl_inline.cuh"

  USL::USL(
      ParticlesContainer _particles,
      NodesContainer _nodes,
      cpu_array<MaterialType> _materials,
      cpu_array<BoundaryConditionType> _boundaryconditions,
      Real _alpha) : Solver(_particles, _nodes, _materials, _boundaryconditions)
  {
    alpha = _alpha;
  }

  /**
   * @brief Reset the temporary arrays for the USL solver
   *
   */
  void USL::reset()
  {
    nodes.reset();
    particles.reset();
    particles.spatial.reset();
  }

  /**
   * @brief Main loop of the USL solver
   *
   */
  void USL::solve()
  {
    reset();

    particles.partition();

    calculate_shape_function(
        nodes,
        particles);

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

    G2P();

    stress_update();

    for (int bc_id = 0; bc_id < boundaryconditions.size(); bc_id++)
    {
      std::visit([&](auto &arg)
                 { arg.apply_on_particles(particles); },
                 boundaryconditions[bc_id]);
    }
  }

  /**
   * @brief Particle to Grid (P2G) operation for USL (velocities gather)
   *
   */
  void USL::P2G()
  {

#ifdef CUDA_ENABLED
    KERNELS_USL_P2G<<<nodes.launch_config.tpb,
                      nodes.launch_config.bpg>>>(
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
        nodes.num_nodes, nodes.inv_node_spacing, nodes.num_nodes_total);
    gpuErrchk(cudaDeviceSynchronize());
#else
    for (size_t ti = 0; ti < nodes.num_nodes_total; ti++)
    {

      usl_p2g_kernel(nodes.moments_gpu.data(),
                     nodes.forces_internal_gpu.data(),
                     nodes.masses_gpu.data(),
                     nodes.node_ids_gpu.data(),
                     particles.stresses_gpu.data(),
                     particles.forces_external_gpu.data(),
                     particles.velocities_gpu.data(),
                     particles.dpsi_gpu.data(),
                     particles.psi_gpu.data(),
                     particles.masses_gpu.data(),
                     particles.volumes_gpu.data(),
                     particles.spatial.cell_start_gpu.data(),
                     particles.spatial.cell_end_gpu.data(),
                     particles.spatial.sorted_index_gpu.data(),
                     particles.is_rigid_gpu.data(),
                     nodes.num_nodes,
                     nodes.inv_node_spacing,
                     nodes.num_nodes_total,
                     ti);
    }
#endif
  }

  /**
   * @brief Grid to Particle (G2P) operation for USL (velocities scatter)
   *
   */
  void USL::G2P()
  {
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
        thrust::raw_pointer_cast(particles.masses_gpu.data()),
        thrust::raw_pointer_cast(particles.is_rigid_gpu.data()),
        thrust::raw_pointer_cast(nodes.moments_gpu.data()),
        thrust::raw_pointer_cast(nodes.moments_nt_gpu.data()),
        thrust::raw_pointer_cast(nodes.masses_gpu.data()),
        particles.spatial.num_cells, particles.num_particles, alpha);
    gpuErrchk(cudaDeviceSynchronize());
#else
    for (size_t ti = 0; ti < particles.num_particles; ti++)
    {
      usl_g2p_kernel(
          particles.velocity_gradient_gpu.data(),
          particles.F_gpu.data(),
          particles.velocities_gpu.data(),
          particles.positions_gpu.data(),
          particles.volumes_gpu.data(),
          particles.dpsi_gpu.data(),
          particles.spatial.bins_gpu.data(),
          particles.volumes_original_gpu.data(),
          particles.psi_gpu.data(),
          particles.masses_gpu.data(),
          particles.is_rigid_gpu.data(),
          nodes.moments_gpu.data(),
          nodes.moments_nt_gpu.data(),
          nodes.masses_gpu.data(),
          particles.spatial.num_cells,
          alpha,
          ti);
    }

#endif
  };
} // namespace pyroclastmpm