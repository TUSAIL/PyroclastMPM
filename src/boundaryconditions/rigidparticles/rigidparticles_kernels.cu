#include "pyroclastmpm/boundaryconditions/rigidparticles/rigidparticles_kernels.cuh"

namespace pyroclastmpm
{

  extern __constant__ Real dt_gpu;
  extern __constant__ int dimension_global_gpu;

  // CALCULATE normals of non-rigid particles
  extern __constant__ int num_surround_nodes_gpu;
  extern __constant__ int backward_window_gpu[64][3];

  __device__ inline Real linear_rigid(const Real relative_distance,
                                      const int node_type = 0)
  {
    double abs_relative_distance = fabs(relative_distance);
    if (abs_relative_distance >= 1.0)
    {
      return 0.0;
    }

    return 1.0 - abs_relative_distance;
  }

  __global__ void KERNELS_CALC_NON_RIGID_GRID_NORMALS(
      Vectorr *grid_normals_gpu,
      const Vectori *node_ids_gpu,
      const Vectorr *particles_dpsi_gpu,
      const Real *particles_masses_gpu,
      const int *particles_cells_start_gpu,
      const int *particles_cells_end_gpu,
      const int *particles_sorted_indices_gpu,
      const Vectori num_nodes,
      const int num_nodes_total)
  {
    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    Vectori node_bin = node_ids_gpu[node_mem_index];

    Vectorr normal = Vectorr::Zero();

    /* loop over non-rigid material points. Use shape function defined there*/

    for (int sid = 0; sid < num_surround_nodes_gpu; sid++)
    {

#if DIM == 3

      const Vectori backward_mesh = Vectori({backward_window_gpu[sid][0],
                                             backward_window_gpu[sid][1],
                                             backward_window_gpu[sid][2]});
      const Vectori selected_bin = node_bin + backward_mesh;
      const unsigned int node_hash =
          selected_bin[0] + selected_bin[1] * num_nodes[0] +
          selected_bin[2] * num_nodes[0] * num_nodes[1];

#elif DIM == 2

      const Vectori backward_mesh = Vectori({backward_window_gpu[sid][0],
                                             backward_window_gpu[sid][1]});
      const Vectori selected_bin = node_bin + backward_mesh;
      const unsigned int node_hash =
          selected_bin[0] + selected_bin[1] * num_nodes[0];
#else

      const Vectori backward_mesh = Vectori(backward_window_gpu[sid][0]);
      const Vectori selected_bin = node_bin + backward_mesh;
      const unsigned int node_hash = selected_bin[0];
#endif

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

        const Vectorr dpsi_particle =
            particles_dpsi_gpu[particle_id * num_surround_nodes_gpu + sid];

        normal += dpsi_particle * particles_masses_gpu[particle_id];
      }
    }
    normal.normalize();
    grid_normals_gpu[node_mem_index] = normal;
  }

  __global__ void KERNEL_GET_OVERLAPPING_RIGID_BODY_GRID(
      bool *is_overlapping_gpu,
      const Vectori *node_ids_gpu,
      const Vectorr *particles_positions_gpu,
      const Vectori *particles_bins_gpu,
      const Vectori num_nodes,
      const Vectorr origin,
      const Real inv_cell_size,
      const int num_nodes_total,
      const int num_particles)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_particles)
    {
      return;
    } // block access threads

    const Vectori particle_bin =
        particles_bins_gpu[tid]; // TODO not whole grid has to be computed?

    const Vectorr particle_coords = particles_positions_gpu[tid];

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

#pragma unroll
    for (int sid = 0; sid < 27; sid++)
    {

#if DIM == 3
      const Vectori backward_mesh = Vectori({linear_backward_window_3d[sid][0],
                                             linear_backward_window_3d[sid][1],
                                             linear_backward_window_3d[sid][2]});
      const Vectori selected_bin = particle_bin + backward_mesh;
      const unsigned int node_hash =
          selected_bin[0] + selected_bin[1] * num_nodes[0] +
          selected_bin[2] * num_nodes[0] * num_nodes[1];
#elif DIM == 2
      const Vectori backward_mesh = Vectori({linear_backward_window_3d[sid][0],
                                             linear_backward_window_3d[sid][1]});
      const Vectori selected_bin = particle_bin + backward_mesh;
      const unsigned int node_hash =
          selected_bin[0] + selected_bin[1] * num_nodes[0];
#else
      const Vectori backward_mesh = Vectori(linear_backward_window_3d[sid][0]);
      const Vectori selected_bin = particle_bin + backward_mesh;
      const unsigned int node_hash = selected_bin[0];
#endif

      const Vectorr relative_coordinates =
          (particle_coords - origin) * inv_cell_size - selected_bin.cast<Real>();

      const Real radius = 1.;
      if (fabs(relative_coordinates(0)) >= radius)
      {
        continue;
      }

#if DIM > 1
      if (fabs(relative_coordinates(1)) >= radius)
      {
        continue;
      }
#endif

#if DIM > 2
      if (fabs(relative_coordinates(2)) >= radius)
        continue;
#endif

      if (node_hash >= num_nodes_total)
      {
        continue;
      }

      is_overlapping_gpu[node_hash] = true;
    }
  }

  /**
   * @brief This kernel sets the total force and integrates the force to find
   * the moment.
   *
   * @param nodes_moments_nt_gpu output nodal moments at next incremental step
   * (USL)
   * @param nodes_forces_total_gpu  output total nodal forces
   * @param nodes_forces_external_gpu nodal external forces (gravity)
   * @param nodes_forces_internal_gpu nodal internal forces ( from stress )
   * @param nodes_moments_gpu nodal moment at currentl incremental step
   * @param nodes_masses_gpu nodal mass
   * @param dt time step
   * @param num_nodes number of nodes in grid dimensions
   */
  __global__ void KERNEL_VELOCITY_CORRECTOR(Vectorr *nodes_moments_nt_gpu,
                                            Vectorr *nodes_moments_gpu,
                                            const int *closest_rigid_id_gpu,
                                            const Vectorr *rigid_velocities_gpu,
                                            const Vectori *node_ids_3d_gpu,
                                            const Real *nodes_masses_gpu,
                                            const Vectorr *grid_normals_gpu,
                                            const bool *is_overlapping_gpu,
                                            const Matrixr rotation_matrix,
                                            const Vectorr COM,
                                            const Vectorr translational_velocity,
                                            const Vectorr origin,
                                            const Real inv_cell_size,
                                            const int num_nodes_total)
  {
    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    Real node_mass = nodes_masses_gpu[node_mem_index];

    if (node_mass <= 0.000000001)
    {
      return;
    }

    if (is_overlapping_gpu[node_mem_index] == false)
    {
      return;
    }

    // get body velocityvolumes_original_gpu
    const int nearest_rigid_id = closest_rigid_id_gpu[node_mem_index];

    // Vector3r rotational_velocity = Vectorr::Zero();
    Vectorr body_veloctity = Vectorr::Zero();

    if (nearest_rigid_id > 0)
    {
      // const Vector3r rigid_pos = rigid_positions_gpu[nearest_rigid_id];
      // const Vector3r rigid_pos =
      //     node_ids_3d_gpu[node_mem_index].cast<Real>() * inv_cell_size +
      //     origin;

      // const Vector3r relative_position = rigid_pos - COM;
      // const Vector3r rotated_position =
      //     rotation_matrix * relative_position - relative_position + COM;

      // rotational_velocity = rotated_position / dt_gpu;

      body_veloctity = rigid_velocities_gpu[nearest_rigid_id];
    }

    // get normal
    const Vectorr normal = grid_normals_gpu[node_mem_index];

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

  __global__ void KERNEL_UPDATE_POS_RIGID(Vectorr *particles_positions_gpu,
                                          Vectorr *particles_velocities_gpu,
                                          const Matrixr rotation_matrix,
                                          const Vectorr COM,
                                          const Vectorr translational_velocity,
                                          const int num_particles)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_particles)
    {
      return;
    } // block access threads

    const Vectorr pos = particles_positions_gpu[tid];

    const Vectorr relative_position = pos - COM;

    const Vectorr rotated_position =
        rotation_matrix * relative_position - relative_position + COM;

    const Vectorr position = particles_positions_gpu[tid];
    const Vectorr position_next =
        translational_velocity * dt_gpu ;
        // + rotated_position;
    particles_positions_gpu[tid] = position + position_next;

    particles_velocities_gpu[tid] = position_next / dt_gpu;
  }

  __global__ void KERNEL_FIND_NEAREST_RIGIDPARTICLE(
      int *closest_rigid_id_gpu,
      const Vectorr *rigid_positions_gpu,
      const Vectori *node_ids_gpu,
      const Real *nodes_masses_gpu,
      const int *rigid_cells_start_gpu,
      const int *rigid_cells_end_gpu,
      const int *rigid_sorted_indices_gpu,
      const bool *is_overlapping_gpu,
      const Vectori num_nodes,
      const Vectorr origin,
      const Real inv_cell_size,
      const int num_nodes_total)
  {
    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    // Real node_mass = nodes_masses_gpu[node_mem_index];
    // if (node_mass <= 0.000000001) {
    //   return;
    // }

    // if (is_overlapping_gpu[node_mem_index] == false) {
    // return;
    // }

    const Vectori node_bin = node_ids_gpu[node_mem_index];

    // const Vectorr nodel_coordinates =
    // node_bin.cast<Real>() /inv_cell_size + origin;

    const int linear_backward_window_3d[64][3] = {
        {0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, -1, 0}, {1, -1, 0}, {-1, -1, 0}, {0, 0, 1}, {1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {-1, 1, 1}, {0, -1, 1}, {1, -1, 1}, {-1, -1, 1}, {0, 0, -1}, {1, 0, -1}, {-1, 0, -1}, {0, 1, -1}, {1, 1, -1}, {-1, 1, -1}, {0, -1, -1}, {1, -1, -1}, {-1, -1, -1}};

    Real min_dist = 999999999999999.;
    int min_id = -1;

#pragma unroll
    for (int sid = 0; sid < 27; sid++)
    {

#if DIM == 3
      const Vectori backward_mesh = Vectori({linear_backward_window_3d[sid][0],
                                             linear_backward_window_3d[sid][1],
                                             linear_backward_window_3d[sid][2]});

      const Vectori selected_bin = node_bin + backward_mesh;

      const unsigned int node_hash =
          selected_bin[0] + selected_bin[1] * num_nodes[0] +
          selected_bin[2] * num_nodes[0] * num_nodes[1];

#elif DIM == 2
      const Vectori backward_mesh = Vectori({linear_backward_window_3d[sid][0],
                                             linear_backward_window_3d[sid][1]});

      const Vectori selected_bin = node_bin + backward_mesh;

      const unsigned int node_hash =
          selected_bin[0] + selected_bin[1] * num_nodes[0];

#else

      const Vectori backward_mesh = Vectori(linear_backward_window_3d[sid][0]);

      const Vectori selected_bin = node_bin + backward_mesh;

      const unsigned int node_hash = selected_bin[0];
#endif
          if (node_hash >= num_nodes_total) {
            continue;
          }

          const int cstart = rigid_cells_start_gpu[node_hash];
          const int cend = rigid_cells_end_gpu[node_hash];

          if ((cstart < 0) || (cend < 0)) {
            continue;
          }

          for (int j = cstart; j < cend; j++) {
            const int rigid_particle_id = rigid_sorted_indices_gpu[j];

            const Vectorr relative_pos =
                (rigid_positions_gpu[rigid_particle_id] - origin) * inv_cell_size -
                selected_bin.cast<Real>();

            // const Vectorr relative_pos =
            // rigid_positions_gpu[rigid_particle_id] - nodel_coordinates;

            const Real distance =
                relative_pos.dot(relative_pos);  // square root function is monotonic

            if (distance < min_dist) {
              min_dist = distance;
              // min_id = rigid_particle_id;
              min_id = rigid_particle_id;
            }

          }  // end loop over particles
    } // end loop over neighboring nodes

      closest_rigid_id_gpu[node_mem_index] = min_id;
  }

} // namespace pyroclastmpm
