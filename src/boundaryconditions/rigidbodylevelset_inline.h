__device__ __host__ inline void update_rigid_velocity(
    Vectorr *particles_velocities_gpu, const Vectorr *particles_positions_gpu,
    const bool *particle_is_rigid_gpu, const Vectorr body_velocity,
    const Vectorr COM, const Vectorr angular_velocities,
    const Vectorr euler_angles, const int tid) {

  if (!particle_is_rigid_gpu[tid]) {
    return;
  }

  Vectorr omega = Vectorr::Zero();
#if DIM == 3

  const Real theta = euler_angles[0];
  const Real phi = euler_angles[1];
  const Real psi = euler_angles[2];

  const Real dtheta = angular_velocities[0];
  const Real dphi = angular_velocities[1];
  const Real dpsi = angular_velocities[2];
  omega[0] = dphi * sin(theta) * sin(psi) + dtheta * cos(psi);
  omega[1] = dphi * sin(theta) * cos(psi) - dtheta * sin(psi);
  omega[2] = dphi * cos(theta) + dpsi;

  const Vectorr rotational_velocity =
      omega.cross(particles_positions_gpu[tid] - COM);

  particles_velocities_gpu[tid] = body_velocity + rotational_velocity;
#else
  printf("Rotation problems with 1D and 2d not implemented \n");
#endif
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_UPDATE_RIGID_VELOCITY(
    Vectorr *particles_velocities_gpu, const Vectorr *particles_positions_gpu,
    const bool *particle_is_rigid_gpu, const Vectorr body_velocity,
    const Vectorr COM, const Vectorr angular_velocities,
    const Vectorr euler_angles, const int num_particles) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  update_rigid_velocity(particles_velocities_gpu, particles_positions_gpu,
                        particle_is_rigid_gpu, body_velocity, COM,
                        angular_velocities, euler_angles, tid);
}
#endif

__device__ __host__ inline void
update_rigid_position(Vectorr *particles_positions_gpu,
                      const Vectorr *particles_velocities_gpu,
                      const bool *particle_is_rigid_gpu, const int tid) {

  if (!particle_is_rigid_gpu[tid]) {
    return;
  }

#ifdef CUDA_ENABLED
  particles_positions_gpu[tid] += particles_velocities_gpu[tid] * dt_gpu;
#else
  particles_positions_gpu[tid] += particles_velocities_gpu[tid] * dt_cpu;
#endif
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_UPDATE_RIGID_POSITION(
    Vectorr *particles_positions_gpu, const Vectorr *particles_velocities_gpu,
    const bool *particle_is_rigid_gpu, const int num_particles) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  update_rigid_position(particles_positions_gpu, particles_velocities_gpu,
                        particle_is_rigid_gpu, tid);
}
#endif

__device__ __host__ inline void calculate_grid_normals_nn_rigid(
    Vectorr *nodes_moments_gpu, Vectorr *nodes_moments_nt_gpu,
    const Vectori *node_ids_gpu, const Real *nodes_masses_gpu,
    const bool *is_overlapping_gpu, const Vectorr *particles_velocities_gpu,
    const Vectorr *particles_dpsi_gpu, const Real *particles_masses_gpu,
    const int *particles_cells_start_gpu, const int *particles_cells_end_gpu,
    const int *particles_sorted_indices_gpu, const bool *particle_is_rigid_gpu,
    const Vectorr *particles_positions_gpu, const Vectorr origin,
    const Real inv_cell_size, const Vectori num_nodes,
    const int num_nodes_total, const int node_mem_index) {

  if (!is_overlapping_gpu[node_mem_index]) {
    return;
  }

  const Real node_mass = nodes_masses_gpu[node_mem_index];

  if (node_mass <= 0.000000001) {
    return;
  }

  Vectori node_bin = node_ids_gpu[node_mem_index];

  Vectorr normal = Vectorr::Zero();

  Real min_dist = 999999999999999.;
  int min_id = -1;

#ifdef CUDA_ENABLED
  const int num_surround_nodes = num_surround_nodes_gpu;
#else
  const int num_surround_nodes = num_surround_nodes_cpu;
#endif

/* loop over non-rigid material points. Use shape function defined there*/
#pragma unroll
  for (int sid = 0; sid < num_surround_nodes; sid++) {

#ifdef CUDA_ENABLED
    const Vectori selected_bin = WINDOW_BIN(node_bin, backward_window_gpu, sid);
#else
    const Vectori selected_bin = WINDOW_BIN(node_bin, backward_window_cpu, sid);
#endif

    const unsigned int node_hash = NODE_MEM_INDEX(selected_bin, num_nodes);

    if (node_hash >= num_nodes_total) {
      continue;
    }

    const int cstart = particles_cells_start_gpu[node_hash];
    const int cend = particles_cells_end_gpu[node_hash];

    if ((cstart < 0) || (cend < 0)) {
      continue;
    }

    for (int j = cstart; j < cend; j++) {
      const int particle_id = particles_sorted_indices_gpu[j];
      // if rigid particle find the nearest (real) particle
      // otherwise calculate the grid normal of non rigid particles
      // two functions within same kernel
      if (particle_is_rigid_gpu[particle_id]) {

        const Vectorr relative_pos =
            (particles_positions_gpu[particle_id] - origin) * inv_cell_size -
            selected_bin.cast<Real>();
        const Real distance =
            relative_pos.dot(relative_pos); // squared distance is monotonic, so
                                            // no need to compute
        if (distance < min_dist) {
          min_dist = distance;
          min_id = particle_id;
        }
      } else {
        const Vectorr dpsi_particle =
            particles_dpsi_gpu[particle_id * num_surround_nodes + sid];

        normal += dpsi_particle * particles_masses_gpu[particle_id];
      }
    }
  }

  Vectorr body_veloctity = Vectorr::Zero();
  if (min_id != -1) {
    body_veloctity = particles_velocities_gpu[min_id];
  }

  normal.normalize();

  // get current node velocity
  const Vectorr node_velocity = nodes_moments_gpu[node_mem_index] / node_mass;

  const Vectorr node_velocity_nt =
      nodes_moments_nt_gpu[node_mem_index] / node_mass;

  // get contact velocity
  const Vectorr contact_vel =
      node_velocity - body_veloctity; // insert velocity here

  const Vectorr contact_vel_nt =
      node_velocity_nt - body_veloctity; // insert velocity here

  // establish contact
  const Real release_criteria = contact_vel.dot(normal);

  const Real release_criteria_nt = contact_vel_nt.dot(normal);

  // apply contact
  if (release_criteria >= 0) {
    nodes_moments_gpu[node_mem_index] =
        (node_velocity - contact_vel) * node_mass;
  }

  if (release_criteria_nt >= 0) {
    nodes_moments_nt_gpu[node_mem_index] =
        (node_velocity_nt - contact_vel_nt) * node_mass;
  }
}

#ifdef CUDA_ENABLED
__global__ void KERNELS_GRID_NORMALS_AND_NN_RIGID(
    Vectorr *nodes_moments_gpu, Vectorr *nodes_moments_nt_gpu,
    const Vectori *node_ids_gpu, const Real *nodes_masses_gpu,
    const bool *is_overlapping_gpu, const Vectorr *particles_velocities_gpu,
    const Vectorr *particles_dpsi_gpu, const Real *particles_masses_gpu,
    const int *particles_cells_start_gpu, const int *particles_cells_end_gpu,
    const int *particles_sorted_indices_gpu, const bool *particle_is_rigid_gpu,
    const Vectorr *particles_positions_gpu, const Vectorr origin,
    const Real inv_cell_size, const Vectori num_nodes,
    const int num_nodes_total) {
  const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (node_mem_index >= num_nodes_total) {
    return;
  }

  calculate_grid_normals_nn_rigid(
      nodes_moments_gpu, nodes_moments_nt_gpu, node_ids_gpu, nodes_masses_gpu,
      is_overlapping_gpu, particles_velocities_gpu, particles_dpsi_gpu,
      particles_masses_gpu, particles_cells_start_gpu, particles_cells_end_gpu,
      particles_sorted_indices_gpu, particle_is_rigid_gpu,
      particles_positions_gpu, origin, inv_cell_size, num_nodes,
      num_nodes_total, node_mem_index);
}

#endif

__device__ __host__ inline void get_overlapping_rigid_body_grid(
    bool *is_overlapping_gpu, const Vectori *node_ids_gpu,
    const Vectorr *particles_positions_gpu, const Vectori *particles_bins_gpu,
    const bool *particle_is_rigid_gpu, const Vectori num_nodes,
    const Vectorr origin, const Real inv_cell_size, const int num_nodes_total,
    const int tid) {

  if (!particle_is_rigid_gpu[tid]) // block non-rigid particles
  {
    return;
  }
  const Vectori particle_bin = particles_bins_gpu[tid];

  const Vectorr particle_coords = particles_positions_gpu[tid];

  // We only search in a cube of 3x3x3 nodes around the particle
  const int linear_backward_window_3d[64][3] = {
      {0, 0, 0},   {1, 0, 0},   {-1, 0, 0},  {0, 1, 0},   {1, 1, 0},
      {-1, 1, 0},  {0, -1, 0},  {1, -1, 0},  {-1, -1, 0}, {0, 0, 1},
      {1, 0, 1},   {-1, 0, 1},  {0, 1, 1},   {1, 1, 1},   {-1, 1, 1},
      {0, -1, 1},  {1, -1, 1},  {-1, -1, 1}, {0, 0, -1},  {1, 0, -1},
      {-1, 0, -1}, {0, 1, -1},  {1, 1, -1},  {-1, 1, -1}, {0, -1, -1},
      {1, -1, -1}, {-1, -1, -1}};

#pragma unroll
  for (int sid = 0; sid < 27; sid++) {

#ifdef CUDA_ENABLED
    const Vectori selected_bin =
        WINDOW_BIN(particle_bin, linear_backward_window_3d, sid);
#else
    const Vectori selected_bin =
        WINDOW_BIN(particle_bin, linear_backward_window_3d, sid);
#endif
    const unsigned int node_hash = NODE_MEM_INDEX(selected_bin, num_nodes);
    const Vectorr relative_coordinates =
        (particle_coords - origin) * inv_cell_size - selected_bin.cast<Real>();

    const Real radius = 1.;
    if (fabs(relative_coordinates(0)) >= radius) {
      continue;
    }

#if DIM > 1
    if (fabs(relative_coordinates(1)) >= radius) {
      continue;
    }
#endif

#if DIM > 2
    if (fabs(relative_coordinates(2)) >= radius)
      continue;
#endif

    if (node_hash >= num_nodes_total) {
      continue;
    }

    is_overlapping_gpu[node_hash] = true;
  }
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_GET_OVERLAPPING_RIGID_BODY_GRID(
    bool *is_overlapping_gpu, const Vectori *node_ids_gpu,
    const Vectorr *particles_positions_gpu, const Vectori *particles_bins_gpu,
    const bool *particle_is_rigid_gpu, const Vectori num_nodes,
    const Vectorr origin, const Real inv_cell_size, const int num_nodes_total,
    const int num_particles) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  get_overlapping_rigid_body_grid(is_overlapping_gpu, node_ids_gpu,
                                  particles_positions_gpu, particles_bins_gpu,
                                  particle_is_rigid_gpu, num_nodes, origin,
                                  inv_cell_size, num_nodes_total, tid);
}

#endif