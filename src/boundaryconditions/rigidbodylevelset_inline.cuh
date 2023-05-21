

__global__ void KERNELS_GRID_NORMALS_AND_NN_RIGID(
    Vectorr *nodes_moments_gpu,
    Vectorr *nodes_moments_nt_gpu,
    const Vectori *node_ids_gpu,
    const Real *nodes_masses_gpu,
    const Vectorr *particles_velocities_gpu,
    const Vectorr *particles_dpsi_gpu,
    const Real *particles_masses_gpu,
    const int *particles_cells_start_gpu,
    const int *particles_cells_end_gpu,
    const int *particles_sorted_indices_gpu,
    const bool *particle_is_rigid_gpu,
    const Vectorr *particles_positions_gpu,
    const Vectorr origin,
    const Real inv_cell_size,
    const Vectori num_nodes,
    const int num_nodes_total)
{
    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
        return;
    }
   
    const Real node_mass = nodes_masses_gpu[node_mem_index];

    if (node_mass <= 0.000000001)
    {
      return;
    }

    Vectori node_bin = node_ids_gpu[node_mem_index];

    Vectorr normal = Vectorr::Zero();

    Real min_dist = 999999999999999.;
    int min_id = -1;


    const int linear_backward_window_3d[64][3] = {
        {0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, -1, 0}, {1, -1, 0}, {-1, -1, 0},

        {0, 0, 1},
        {1, 0, 1},
        {-1, 0, 1},
        {0, 1, 1},
        {1, 1, 1},
        {-1, 1, 1},
        {0, -1, 1},
        {1, -1, 1},
        {-1, -1, 1},
        {0, 0, -1},
        {1, 0, -1},
        {-1, 0, -1},
        {0, 1, -1},
        {1, 1, -1},
        {-1, 1, -1},
        {0, -1, -1},
        {1, -1, -1},
        {-1, -1, -1}
    };

// #ifdef CUDA_ENABLED
//     const int num_surround_nodes = num_surround_nodes_gpu;
// #else
//     const int num_surround_nodes = num_surround_nodes_cpu;
// #endif

/* loop over non-rigid material points. Use shape function defined there*/
#pragma unroll
    for (int sid = 0; sid < 27; sid++)
    {

#ifdef CUDA_ENABLED
        const Vectori selected_bin = WINDOW_BIN(node_bin, linear_backward_window_3d, sid);
#else
        const Vectori selected_bin = WINDOW_BIN(node_bin, linear_backward_window_3d, sid);
#endif

        const unsigned int node_hash = NODE_MEM_INDEX(selected_bin, num_nodes);
        
        if (node_hash >= num_nodes_total)
        {
            continue;
        }

        const int cstart = particles_cells_start_gpu[node_hash];
        const int cend = particles_cells_end_gpu[node_hash];

        if ((cstart < 0) || (cend < 0))
        {
            continue;
        }

        for (int j = cstart; j < cend; j++)
        {
            const int particle_id = particles_sorted_indices_gpu[j];
            // if rigid particle find the nearest one
            // otherwise calculate the grid normal of non rigid particles
            // two functions within same kernel
            if (particle_is_rigid_gpu[particle_id])
            {

                const Vectorr relative_pos = (particles_positions_gpu[particle_id] - origin) * inv_cell_size - selected_bin.cast<Real>();
                const Real distance = relative_pos.dot(relative_pos); // squared distance is monotonic, so no need to compute
                if (distance < min_dist) {
                    min_dist = distance;
                    min_id = particle_id;
                }
            } else {
                const Vectorr dpsi_particle =
                particles_dpsi_gpu[particle_id * 27 + sid];

                normal += dpsi_particle * particles_masses_gpu[particle_id];
            }


        }
    }

    if (min_id == -1) {
        return;
    }

    normal.normalize();
    
    // get velocity of nearest rigid particle
    const Vectorr body_veloctity = particles_velocities_gpu[min_id];

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
    if (release_criteria >= 0)
    {
      nodes_moments_gpu[node_mem_index] =
          (node_velocity - contact_vel) * node_mass;
    }

    if (release_criteria_nt >= 0)
    {
      nodes_moments_nt_gpu[node_mem_index] =
          (node_velocity_nt - contact_vel_nt) * node_mass;
    }
}