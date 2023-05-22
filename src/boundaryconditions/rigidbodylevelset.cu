#include "pyroclastmpm/boundaryconditions/rigidbodylevelset.cuh"

namespace pyroclastmpm
{

#ifdef CUDA_ENABLED
  extern __constant__ Real dt_gpu;
  extern __constant__ int num_surround_nodes_gpu;
  extern __constant__ int forward_window_gpu[64][3];
  extern __constant__ int backward_window_gpu[64][3];
#else
  extern int num_surround_nodes_cpu;
  extern int forward_window_cpu[64][3];
  extern int backward_window_cpu[64][3];
#endif

  extern int global_step_cpu;

  extern Real dt_cpu;

// include private header with kernels here to inline them
#include "rigidbodylevelset_inline.cuh"

  RigidBodyLevelSet::RigidBodyLevelSet(const cpu_array<int> _frames,
                                       const cpu_array<Vectorr> _locations,
                                       const cpu_array<Vectorr> _rotations,
                                       const cpu_array<OutputType> _output_formats) : output_formats(_output_formats),
                                                                                      frames_cpu(_frames),
                                                                                      locations_cpu(_locations),
                                                                                      rotations_cpu(_rotations)
  {
    num_frames = _frames.size();

    if (DIM != 3)
    {
      printf("Rigid body level set only supports 3D simulations\n");
      exit(1);
    }

    // COM = Vectorr::Zero(); // initial Center of mass
    // for (int pid = 0; pid < num_particles; pid++)
    // {
    //     COM += _positions[pid];
    // }
    // COM /= num_particles;
    // translational_velocity = Vectorr::Zero(); // initial translational_velocity
    // ROT = Vectorr::Zero();                    // initial euler angles
    // rotation_matrix = Matrixr::Zero();        // initial rotation matrix
  }
  void RigidBodyLevelSet::apply_on_nodes_moments(NodesContainer &nodes_ref,
                                                 ParticlesContainer &particles_ref)
  {

    if (global_step_cpu == 0)
    {
      initialize(nodes_ref, particles_ref);
    }

    // 4. Mask grid nodes that do not contribute to rigid body contact
    calculate_overlapping_rigidbody(nodes_ref, particles_ref);

    // 5. Get rigid body grid normals
    calculate_grid_normals(nodes_ref, particles_ref);
  };

  void RigidBodyLevelSet::initialize(NodesContainer &nodes_ref,
                                     ParticlesContainer &particles_ref)
  {
    set_default_device<Vectorr>(nodes_ref.num_nodes_total, {}, normals_gpu, Vectorr::Zero());
    set_default_device<bool>(nodes_ref.num_nodes_total, {}, is_overlapping_gpu, false);
    set_default_device<int>(nodes_ref.num_nodes_total, {}, closest_rigid_particle_gpu, -1);
  }

  void RigidBodyLevelSet::calculate_grid_normals(
      NodesContainer &nodes_ref,
      ParticlesContainer &particles_ref)
  {

#ifdef CUDA_ENABLED
    KERNELS_GRID_NORMALS_AND_NN_RIGID<<<nodes_ref.launch_config.tpb,
                                        nodes_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(nodes_ref.moments_gpu.data()),
        thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
        thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
        thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()),
        thrust::raw_pointer_cast(is_overlapping_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.velocities_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.dpsi_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.spatial.cell_start_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.spatial.cell_end_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.spatial.sorted_index_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.is_rigid_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
        nodes_ref.node_start, nodes_ref.inv_node_spacing,
        particles_ref.spatial.num_cells, particles_ref.spatial.num_cells_total);
    gpuErrchk(cudaDeviceSynchronize());
#else
    for (int nid = 0; nid < nodes_ref.num_nodes_total; nid++)
    {
      calculate_grid_normals_nn_rigid(
          nodes_ref.moments_gpu.data(),
          nodes_ref.moments_nt_gpu.data(),
          nodes_ref.node_ids_gpu.data(),
          nodes_ref.masses_gpu.data(),
          is_overlapping_gpu.data(),
          particles_ref.velocities_gpu.data(),
          particles_ref.dpsi_gpu.data(),
          particles_ref.masses_gpu.data(),
          particles_ref.spatial.cell_start_gpu.data(),
          particles_ref.spatial.cell_end_gpu.data(),
          particles_ref.spatial.sorted_index_gpu.data(),
          particles_ref.is_rigid_gpu.data(),
          particles_ref.positions_gpu.data(),
          nodes_ref.node_start, nodes_ref.inv_node_spacing,
          particles_ref.spatial.num_cells, particles_ref.spatial.num_cells_total,
          nid);
    }
#endif
  }

  void RigidBodyLevelSet::calculate_overlapping_rigidbody(
      NodesContainer &nodes_ref,
      ParticlesContainer &particles_ref)
  {
    #ifdef CUDA_ENABLED
    KERNEL_GET_OVERLAPPING_RIGID_BODY_GRID<<<particles_ref.launch_config.tpb,
                                             particles_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(is_overlapping_gpu.data()),
        thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.spatial.bins_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.is_rigid_gpu.data()),
        particles_ref.spatial.num_cells,
        particles_ref.spatial.grid_start,
        particles_ref.spatial.inv_cell_size,
        particles_ref.spatial.num_cells_total,
        particles_ref.num_particles);

      gpuErrchk(cudaDeviceSynchronize());
    #else
    for (int pid = 0; pid < particles_ref.num_particles; pid++)
    {
        get_overlapping_rigid_body_grid(
            is_overlapping_gpu.data(),
            nodes_ref.node_ids_gpu.data(),
            particles_ref.positions_gpu.data(),
            particles_ref.spatial.bins_gpu.data(),
            particles_ref.is_rigid_gpu.data(),
            particles_ref.spatial.num_cells,
            particles_ref.spatial.grid_start,
            particles_ref.spatial.inv_cell_size,
            particles_ref.spatial.num_cells_total,
            pid);
    }
    #endif
  }

} // namespace pyroclastmpm