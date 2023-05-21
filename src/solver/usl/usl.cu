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

  __device__ __host__ void inline usl_p2g_kernel(Vectorr *nodes_moments_gpu,
                                                 Vectorr *nodes_forces_internal_gpu,
                                                 Real *nodes_masses_gpu,
                                                 const Vectori *node_ids_gpu,
                                                 const Matrix3r *particles_stresses_gpu,
                                                 const Vectorr *particles_forces_external_gpu,
                                                 const Vectorr *particles_velocities_gpu,
                                                 const Vectorr *particles_dpsi_gpu,
                                                 const Real *particles_psi_gpu,
                                                 const Real *particles_masses_gpu,
                                                 const Real *particles_volumes_gpu,
                                                 const int *particles_cells_start_gpu,
                                                 const int *particles_cells_end_gpu,
                                                 const int *particles_sorted_indices_gpu,
                                                 const Vectori num_nodes,
                                                 const Real inv_cell_size,
                                                 const int num_nodes_total,
                                                 const int node_mem_index)
  {
    const Vectori node_bin = node_ids_gpu[node_mem_index];
    Vectorr total_node_moment = Vectorr::Zero();
    Vectorr total_node_force_internal = Vectorr::Zero();
    Vectorr total_node_force_external = Vectorr::Zero();
    Real total_node_mass = 0.;

#ifdef CUDA_ENABLED
    const int num_surround_nodes = num_surround_nodes_gpu;
#else
    const int num_surround_nodes = num_surround_nodes_cpu;
#endif

#pragma unroll
    for (int sid = 0; sid < num_surround_nodes; sid++)
    {
#ifdef CUDA_ENABLED
      const Vectori selected_bin = WINDOW_BIN(node_bin, backward_window_gpu, sid);
#else
      const Vectori selected_bin = WINDOW_BIN(node_bin, backward_window_cpu, sid);
#endif

      const unsigned int node_hash = NODE_MEM_INDEX(selected_bin, num_nodes);
      if (node_hash >= num_nodes_total)
      {
        continue;
      }

      const int cstart = particles_cells_start_gpu[node_hash];
      if (cstart < 0)
      {
        continue;
      }
      const int cend = particles_cells_end_gpu[node_hash];
      if (cend < 0)
      {
        continue;
      }

      for (int j = cstart; j < cend; j++)
      {
        const int particle_id = particles_sorted_indices_gpu[j];
        const Real psi_particle = particles_psi_gpu[particle_id * num_surround_nodes + sid];
        const Vectorr dpsi_particle = particles_dpsi_gpu[particle_id * num_surround_nodes + sid];
        const Real scaled_mass = psi_particle * particles_masses_gpu[particle_id];
        total_node_mass += scaled_mass;
        total_node_moment += scaled_mass * particles_velocities_gpu[particle_id];
        total_node_force_external += psi_particle * particles_forces_external_gpu[particle_id];
#if DIM == 3
        total_node_force_internal += -1. * particles_volumes_gpu[particle_id] * particles_stresses_gpu[particle_id] * dpsi_particle;
#else
        Matrix3r cauchy_stress_3d = particles_stresses_gpu[particle_id];
        Matrixr cauchy_stress = cauchy_stress_3d.block(0, 0, DIM, DIM);
        total_node_force_internal += -1. * particles_volumes_gpu[particle_id] * cauchy_stress * dpsi_particle;
#endif
      }
    }

    nodes_masses_gpu[node_mem_index] = total_node_mass;
    nodes_moments_gpu[node_mem_index] = total_node_moment;
    nodes_forces_internal_gpu[node_mem_index] = total_node_force_internal + total_node_force_external;
  }

#ifdef CUDA_ENABLED
  __global__ void KERNELS_USL_P2G(Vectorr *nodes_moments_gpu,
                                  Vectorr *nodes_forces_internal_gpu,
                                  Real *nodes_masses_gpu,
                                  const Vectori *node_ids_gpu,
                                  const Matrix3r *particles_stresses_gpu,
                                  const Vectorr *particles_forces_external_gpu,
                                  const Vectorr *particles_velocities_gpu,
                                  const Vectorr *particles_dpsi_gpu,
                                  const Real *particles_psi_gpu,
                                  const Real *particles_masses_gpu,
                                  const Real *particles_volumes_gpu,
                                  const int *particles_cells_start_gpu,
                                  const int *particles_cells_end_gpu,
                                  const int *particles_sorted_indices_gpu,
                                  const Vectori num_nodes,
                                  const Real inv_cell_size,
                                  const int num_nodes_total)
  {

    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    usl_p2g_kernel(nodes_moments_gpu,
                   nodes_forces_internal_gpu,
                   nodes_masses_gpu,
                   node_ids_gpu,
                   particles_stresses_gpu,
                   particles_forces_external_gpu,
                   particles_velocities_gpu,
                   particles_dpsi_gpu,
                   particles_psi_gpu,
                   particles_masses_gpu,
                   particles_volumes_gpu,
                   particles_cells_start_gpu,
                   particles_cells_end_gpu,
                   particles_sorted_indices_gpu,
                   num_nodes,
                   inv_cell_size,
                   num_nodes_total,
                   node_mem_index);
  }

#endif
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
                     nodes.num_nodes,
                     nodes.inv_node_spacing,
                     nodes.num_nodes_total,
                     ti);
    }
#endif
  }

  __device__ __host__ inline void usl_g2p_kernel(
      Matrixr *particles_velocity_gradients_gpu,
      Matrixr *particles_F_gpu,
      Vectorr *particles_velocities_gpu,
      Vectorr *particles_positions_gpu,
      Real *particles_volumes_gpu,
      const Vectorr *particles_dpsi_gpu,
      const Vectori *particles_bins_gpu,
      const Real *particles_volumes_original_gpu,
      const Real *particles_psi_gpu,
      const Real *particles_masses_gpu,
      const Vectorr *nodes_moments_gpu,
      const Vectorr *nodes_moments_nt_gpu,
      const Real *nodes_masses_gpu,
      const Vectori num_cells,
      const Real alpha,
      const int tid)
  {

    const Vectori particle_bin = particles_bins_gpu[tid];
    const Vectorr particle_coords = particles_positions_gpu[tid];

    Vectorr vel_inc = Vectorr::Zero();
    Vectorr dvel_inc = Vectorr::Zero();
    Matrixr vel_grad = Matrixr::Zero();

#ifdef CUDA_ENABLED
    const int num_surround_nodes = num_surround_nodes_gpu;
#else
    const int num_surround_nodes = num_surround_nodes_cpu;
#endif

    for (int i = 0; i < num_surround_nodes; i++)
    {

#ifdef CUDA_ENABLED
      const Vectori selected_bin = WINDOW_BIN(particle_bin, forward_window_gpu, i);
#else
      const Vectori selected_bin = WINDOW_BIN(particle_bin, forward_window_cpu, i);
#endif
      bool invalidCell = false;
      const unsigned int nhash = NODE_MEM_INDEX(selected_bin, num_cells);
      // TODO this is slow!
      for (int axis = 0; axis < DIM; axis++)
      {
        if ((selected_bin[axis] < 0) || (selected_bin[axis] >= num_cells[axis]))
        {
          invalidCell = true;
          break;
        }
      }
      if (invalidCell)
      {
        continue;
      }

      const Real node_mass = nodes_masses_gpu[nhash];
      if (node_mass <= 0.000000001)
      {
        continue;
      }

      const Real psi_particle = particles_psi_gpu[tid * num_surround_nodes + i];
      const Vectorr dpsi_particle = particles_dpsi_gpu[tid * num_surround_nodes + i];

      const Vectorr node_velocity = nodes_moments_gpu[nhash] / node_mass;
      const Vectorr node_velocity_nt = nodes_moments_nt_gpu[nhash] / node_mass;
      const Vectorr delta_velocity = node_velocity_nt - node_velocity;

      dvel_inc += psi_particle * delta_velocity;
      vel_inc += psi_particle * node_velocity_nt;
      vel_grad += dpsi_particle * node_velocity_nt.transpose();
    }
    particles_velocities_gpu[tid] = alpha * (particles_velocities_gpu[tid] + dvel_inc) + (1. - alpha) * vel_inc;
    particles_velocity_gradients_gpu[tid] = vel_grad;
#ifdef CUDA_ENABLED
    particles_positions_gpu[tid] = particle_coords + dt_gpu * vel_inc;
    particles_F_gpu[tid] = (Matrixr::Identity() + vel_grad * dt_gpu) * particles_F_gpu[tid];
#else
    particles_positions_gpu[tid] = particle_coords + dt_cpu * vel_inc;
    particles_F_gpu[tid] = (Matrixr::Identity() + vel_grad * dt_cpu) * particles_F_gpu[tid];
#endif
    Real J = particles_F_gpu[tid].determinant();
    particles_volumes_gpu[tid] = J * particles_volumes_original_gpu[tid];
  }

#ifdef CUDA_ENABLED
  __global__ void KERNEL_USL_G2P(Matrixr *particles_velocity_gradients_gpu,
                                 Matrixr *particles_F_gpu,
                                 Vectorr *particles_velocities_gpu,
                                 Vectorr *particles_positions_gpu,
                                 Real *particles_volumes_gpu,
                                 const Vectorr *particles_dpsi_gpu,
                                 const Vectori *particles_bins_gpu,
                                 const Real *particles_volumes_original_gpu,
                                 const Real *particles_psi_gpu,
                                 const Real *particles_masses_gpu,
                                 const Vectorr *nodes_moments_gpu,
                                 const Vectorr *nodes_moments_nt_gpu,
                                 const Real *nodes_masses_gpu,
                                 const Vectori num_cells,
                                 const int num_particles,
                                 const Real alpha)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_particles)
    {
      return;
    } // block access threads

    usl_g2p_kernel(particles_velocity_gradients_gpu,
                   particles_F_gpu,
                   particles_velocities_gpu,
                   particles_positions_gpu,
                   particles_volumes_gpu,
                   particles_dpsi_gpu,
                   particles_bins_gpu,
                   particles_volumes_original_gpu,
                   particles_psi_gpu,
                   particles_masses_gpu,
                   nodes_moments_gpu,
                   nodes_moments_nt_gpu,
                   nodes_masses_gpu,
                   num_cells,
                   alpha,
                   tid);
  }
#endif
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