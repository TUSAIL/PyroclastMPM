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

#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern __constant__ Real dt_gpu;
extern __constant__ int num_surround_nodes_gpu;
extern __constant__ int g2p_window_gpu[64][3];
extern __constant__ int p2g_window_gpu[64][3];
extern Real __constant__ dt_gpu;
#else
extern const int num_surround_nodes_cpu;
extern const int g2p_window_cpu[64][3];
extern const int p2g_window_cpu[64][3];
extern const Real dt_cpu;
#endif

/**
 * @brief Update velocity of non rigid bodies
 * @details The total velocity of a non rigid body is the sum of its
 * translational and rotational velocity
 *
 * @param particles_velocities_gpu velocity of the particles
 * @param particles_positions_gpu positions of the particles
 * @param particle_is_rigid_gpu boolean array to check if particle is rigid
 * @param body_velocity translational velocity of the body
 * @param COM center of mass of the body
 * @param angular_velocities angular velocity of the body
 * @param euler_angles euler angles of the body
 * @param tid particle id
 */
__device__ __host__ inline void update_rigid_velocity(
    Vectorr *particles_velocities_gpu, const Vectorr *particles_positions_gpu,
    const bool *particle_is_rigid_gpu, const Vectorr body_velocity,
    const Vectorr COM, const Vectorr angular_velocities,
    const Vectorr euler_angles, const int tid) {

  if (!particle_is_rigid_gpu[tid]) {
    return;
  }



  // x-to y, y-to z, z-to x
  // roll, pitch, yaw
  // const Real theta = euler_angles[0]; // about x axis
  // const Real phi = euler_angles[1]; // about y axis
  // const Real psi = euler_angles[2]; // about z axis

  // const Real dtheta = angular_velocities[0];
  // const Real dphi = angular_velocities[1];
  // const Real dpsi = angular_velocities[2];

  // Vector3r omega = Vectorr::Zero();
  // omega[0] = dphi * sin(theta) * sin(psi) + dtheta * cos(psi);
  // omega[2] = dphi * sin(theta) * cos(psi) - dtheta * sin(psi);
  // omega[1] = dphi * cos(theta) + dpsi;

  

  // const Vector3r rotational_velocity =
  //     omega.cross(particles_positions_gpu[tid] - COM);

  // printf("rotational_velocity %f %f %f \n", rotational_velocity[0],
  //        rotational_velocity[1], rotational_velocity[2]);


  // printf("body_velocity %f %f %f \n", body_velocity[0], body_velocity[1],body_velocity[2]);

  // // printf("rotational_velocity %f %f %f \n", rotational_velocity[0],
  // //        rotational_velocity[1], rotational_velocity[2]);

  // // particles_velocities_gpu[tid] = body_velocity + rotational_velocity;
  // particles_velocities_gpu[tid] = body_velocity;

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

/**
 * @brief Update position of rigid particles
 *
 * @param particles_positions_gpu position of the particles
 * @param particles_velocities_gpu velocity of the particles
 * @param particle_is_rigid_gpu boolean array to check if particle is rigid
 * @param tid
 * @return __device__
 */
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

/**
 * @brief  Calculate the grid normal and find the nearest rigid particle
 * @param nodes_moments_gpu Node moments
 * @param nodes_moments_nt_gpu Forward node moments
 * @param node_ids_gpu node bin ids (idx,idy,idz)
 * @param nodes_masses_gpu node masses
 * @param is_overlapping_gpu a boolean array if rigid body and non rigidy body
 * share the same nodes
 * @param particles_velocities_gpu velocity of the particles
 * @param particles_dpsi_gpu shape function gradient of the particles
 * @param particles_masses_gpu mass of the particles
 * @param particles_cells_start_gpu start index of the particles in the cell
 * @param particles_cells_end_gpu end index of the particles in the cell
 * @param particles_sorted_indices_gpu sorted indices of the particles
 * @param particle_is_rigid_gpu boolean array to check if particle is rigid
 * @param particles_positions_gpu position of the particles
 * @param origin origin of the grid
 * @param inv_cell_size inverse of the cell size
 * @param num_nodes number of nodes in the grid
 * @param num_nodes_total total number of nodes
 * @param node_mem_index
 * @return
 */
__device__ __host__ inline void calculate_grid_normals_nn_rigid(
    Vectorr *nodes_normals_gpu, Vectorr *nodes_moments_gpu,
    Vectorr *nodes_moments_nt_gpu, const Vectori *node_ids_gpu,
    const Real *nodes_masses_gpu, bool *is_overlapping_gpu,
    const Vectorr *particles_velocities_gpu, const Vectorr *particles_dpsi_gpu,
    const Real *particles_masses_gpu, const int *particles_cells_start_gpu,
    const int *particles_cells_end_gpu, const int *particles_sorted_indices_gpu,
    const bool *particle_is_rigid_gpu, const Vectorr *particles_positions_gpu,
    const Grid &grid, const int node_mem_index) {

// #if DIM > 2 // TODO remove this
  // if (!is_overlapping_gpu[node_mem_index]) {
  //   return;
  // }

  // printf("rigid node found?");
  const Real node_mass = nodes_masses_gpu[node_mem_index];

  // if (node_mem_index == 400) {
  //   printf("mass %f \n", node_mass);
  // }
  if (node_mass <= 0.000000001) {
    return;
  }

  Vectori node_bin = node_ids_gpu[node_mem_index];

  Vectorr normal = Vectorr::Zero();
  bool has_normal = false;

  Real min_dist = (Real)999999999999999.;
  int min_id = -1;

    #if DIM == 3
  const int linear_p2g_window[64][3] = {{0, 0, 0},   {0, 0, -1},  {-1, 0, 0},
                                      {-1, 0, -1}, {0, -1, 0},  {0, -1, -1},
                                      {-1, -1, 0}, {-1, -1, -1}}
  const int num_surround_nodes = 8;
  #elif DIM == 2
  const int linear_p2g_window[64][3] = {
    {0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {-1, -1, 0}};
  const int num_surround_nodes = 4;
 #else
  const int linear_p2g_window[64][3] = {{0, 0, 0}, {-1, 0, 0}};
  const int num_surround_nodes = 2;
 #endif

// #ifdef CUDA_ENABLED
//   const int num_surround_nodes = num_surround_nodes_gpu;
// #else
//   const int num_surround_nodes = num_surround_nodes_cpu;
// #endif

  // bool has_normal_particle = false;
// loop over non-rigid material to calculate grid normalcalculate_grid_normals_nn_rigid
#pragma unroll
  for (int sid = 0; sid < num_surround_nodes; sid++) {
 const Vectori selected_bin = WINDOW_BIN(node_bin, linear_p2g_window, sid);
#ifdef CUDA_ENABLED
    // const Vectori selected_bin = WINDOW_BIN(node_bin, p2g_window_gpu, sid);
#else
    // const Vectori selected_bin = WINDOW_BIN(node_bin, p2g_window_cpu, sid);
#endif

    const unsigned int node_hash = NODE_MEM_INDEX(selected_bin, grid.num_cells);

    if (node_hash >= grid.num_cells_total) {
      continue;
    }

    const int cstart = particles_cells_start_gpu[node_hash];
    if (cstart < 0) {
      continue;
    }
    const int cend = particles_cells_end_gpu[node_hash];
    if (cend < 0) {
      continue;
    }

    for (int j = cstart; j < cend; j++) {
      const int particle_id = particles_sorted_indices_gpu[j];

      if (particle_is_rigid_gpu[particle_id]) {
        
        const Vectorr relative_pos =
            (particles_positions_gpu[particle_id] - grid.origin) *
                grid.inv_cell_size -
            selected_bin.cast<Real>();
        const Real distance =
            relative_pos.dot(relative_pos); // squared distance is
                                            // monotonic,
        // so no need to compute
        if (distance < min_dist) {
          min_dist = distance;
          min_id = particle_id;
        }

      } else {
        // has_normal_particle = true;
        const Vectorr dpsi_particle =
            particles_dpsi_gpu[particle_id * num_surround_nodes + sid];

        normal += dpsi_particle * particles_masses_gpu[particle_id];

        // needed?
        if (has_normal == false) {
          has_normal = true;
        }
      }
    }
  }
  
  if (min_id == -1)
  {
    // node does not overlap
    return;
  }

  Vectorr body_veloctity = particles_velocities_gpu[min_id];

  // if (!has_normal_particle) {
  //   is_overlapping_gpu[node_mem_index] = false;
  //   return;
  // }

  // Vectorr body_veloctity = Vectorr::Zero();
  // if (min_id != -1) {
    // TODO: Fix this
    // 
  // }

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

// #endif // endif DDIM=2 TODO: Remove this (or fix for 2D)
}

#ifdef CUDA_ENABLED
__global__ void KERNELS_GRID_NORMALS_AND_NN_RIGID(
    Vectorr *nodes_normals_gpu, Vectorr *nodes_moments_gpu,
    Vectorr *nodes_moments_nt_gpu, const Vectori *node_ids_gpu,
    const Real *nodes_masses_gpu, bool *is_overlapping_gpu,
    const Vectorr *particles_velocities_gpu, const Vectorr *particles_dpsi_gpu,
    const Real *particles_masses_gpu, const int *particles_cells_start_gpu,
    const int *particles_cells_end_gpu, const int *particles_sorted_indices_gpu,
    const bool *particle_is_rigid_gpu, const Vectorr *particles_positions_gpu,
    const Grid grid) {
  const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (node_mem_index >= grid.num_cells_total) {
    return;
  }

  calculate_grid_normals_nn_rigid(
      nodes_normals_gpu, nodes_moments_gpu, nodes_moments_nt_gpu, node_ids_gpu,
      nodes_masses_gpu, is_overlapping_gpu, particles_velocities_gpu,
      particles_dpsi_gpu, particles_masses_gpu, particles_cells_start_gpu,
      particles_cells_end_gpu, particles_sorted_indices_gpu,
      particle_is_rigid_gpu, particles_positions_gpu, grid, node_mem_index);
}

#endif

__device__ __host__ inline void g2p_get_nodes_w_rigid_particles(
    bool *is_overlapping_gpu, const Vectorr *particles_positions_gpu,
    const Vectori *particles_bins_gpu, int *particles_cells_start_gpu,
    const int *particles_cells_end_gpu, const int *particles_sorted_indices_gpu,
    const bool *particles_is_rigid_gpu, const Grid &grid, const int nid) {

  // printf("g2p_get_nodes_w_rigid_particles \n");
  // if (!particle_is_rigid_gpu[nid]) {
  //   return;
  // }
  const Vectorr particle_coords = particles_positions_gpu[nid];

  const Vectori particle_bin = particles_bins_gpu[nid];
  
  #if DIM == 3
  const int linear_p2g_window[64][3] = {{0, 0, 0},   {0, 0, -1},  {-1, 0, 0},
                                      {-1, 0, -1}, {0, -1, 0},  {0, -1, -1},
                                      {-1, -1, 0}, {-1, -1, -1}}
  const int num_surround_nodes = 8;
  #elif DIM == 2
  const int linear_p2g_window[64][3] = {
    {0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {-1, -1, 0}};
  const int num_surround_nodes = 4;
 #else
  const int linear_p2g_window[64][3] = {{0, 0, 0}, {-1, 0, 0}};
  const int num_surround_nodes = 2;
 #endif

  bool overlap_rigid = false;
  bool overlap_non_rigid = false;

  for (int sid = 0; sid < num_surround_nodes; sid++) {

    const Vectori selected_bin = WINDOW_BIN(selected_bin, linear_p2g_window, sid);

    const unsigned int node_hash = NODE_MEM_INDEX(selected_bin, grid.num_cells);
    
    if (node_hash >= grid.num_cells_total) {
      continue;
    }

    const int cstart = particles_cells_start_gpu[node_hash];
    if (cstart < 0) {
      continue;
    }
    const int cend = particles_cells_end_gpu[node_hash];
    if (cend < 0) {
      continue;
    }

    for (int j = cstart; j < cend; j++) {

      const int particle_id = particles_sorted_indices_gpu[j];

      if (particles_is_rigid_gpu[particle_id])
      {
        overlap_rigid = true;
      } else {
        overlap_non_rigid = true;
      }

    }
  }

  if (overlap_rigid && overlap_non_rigid) {
    is_overlapping_gpu[nid] = true;
  } else {
    is_overlapping_gpu[nid] = false;
  }

    // const Vectori selected_bin =
    //     WINDOW_BIN(particle_bin, linear_backward_window, i);


// // #ifdef CUDA_ENABLED
// //     const Vectori selected_bin =
// //         WINDOW_BIN(particle_bin, linear_backward_window, i);
// // #else
// //     const Vectori selected_bin =
// //         WINDOW_BIN(particle_bin, linear_backward_window_3d, i);
// // #endif

//     bool invalidCell = false;
//     const unsigned int nhash = NODE_MEM_INDEX(selected_bin, grid.num_cells);

//     const Vectorr relative_coordinates =
//         (particle_coords - grid.origin) * grid.inv_cell_size -
//         selected_bin.cast<Real>();

//     const Real radius = 1;
//     if (fabs(relative_coordinates(0)) >= radius) {
//       continue;
//     }

// #if DIM > 1
//     if (fabs(relative_coordinates(1)) >= radius) {
//       continue;
//     }
// #endif

// #if DIM > 2
//     if (fabs(relative_coordinates(2)) >= radius)
//       continue;
// #endif
//     if (nhash >= grid.num_cells_total) {
//       continue;
//     }
//     is_overlapping_gpu[nhash] = true;

    //   //     if (is_rigid_body) {
    //   //       is_overlapping_gpu[node_hash] = is_overlapping;
    //   //     } else {
    //   //       if (is_overlapping_gpu[node_hash]) {
    //   //         is_overlapping_gpu[node_hash] = is_overlapping;
    //   //       } else {
    //   //         is_overlapping_gpu[node_hash] = false;
    //   //       }
    //   //     }
    //   //     // printf("[%d] overlapping with node %d %d %d \n", tid,
    //   //     selected_bin[0],
    //   //     //  selected_bin[1], selected_bin[2]);
    //   //   }

    // // TODO: this is slow!
    // for (int axis = 0; axis < DIM; axis++) {
    //   if ((selected_bin[axis] < 0) ||
    //       (selected_bin[axis] >= grid.num_cells[axis])) {
    //     invalidCell = true;
    //     break;
    //   }
    // }

    // if (invalidCell) {
    //   continue;
    // }

    // const int cstart = particles_cells_start_gpu[nhash];
    // if (cstart < 0) {
    //   continue;
    // }
    // const int cend = particles_cells_end_gpu[nhash];
    // if (cend < 0) {
    //   continue;
    // }

    // for (int j = cstart; j < cend; j++) {
    //   const int particle_id = particles_sorted_indices_gpu[j];

    //   if (!particle_is_rigid_gpu[particle_id]) {
    //     is_overlapping_gpu[nhash] = true;
    //   }
    // }

    // is_overlapping_gpu[nhash] = true;
  // }

  // __device__ __host__ inline void get_overlapping_rigid_body_grid(
  //     bool *is_overlapping_gpu, const Vectori *node_ids_gpu,
  //     int *particles_cells_start_gpu, const int *particles_cells_end_gpu,
  //     const int *particles_sorted_indices_gpu, const bool
  //     *particle_is_rigid_gpu, const Grid &grid, const int node_mem_index) {

  //   Vectori node_bin = node_ids_gpu[node_mem_index];

  // #ifdef CUDA_ENABLED
  //   const int num_surround_nodes = num_surround_nodes_gpu;
  // #else
  //   const int num_surround_nodes = num_surround_nodes_cpu;
  // #endif
  //   // bool is_overlapping = false;

  //   bool found_rigid = false;
  //   bool found_non_rigid = false;

  // // loop over non-rigid material to calculate grid normal
  // #pragma unroll
  //   for (int sid = 0; sid < num_surround_nodes; sid++) {

  // #ifdef CUDA_ENABLED
  //     const Vectori selected_bin = WINDOW_BIN(node_bin, g2p_window_gpu, sid);
  // #else
  //     const Vectori selected_bin = WINDOW_BIN(node_bin, g2p_window_cpu, sid);
  // #endif

  //     const unsigned int node_hash = NODE_MEM_INDEX(selected_bin,
  //     grid.num_cells);

  //     if (node_hash >= grid.num_cells_total) {
  //       continue;
  //     }

  //     const int cstart = particles_cells_start_gpu[node_hash];

  //     if (cstart < 0) {
  //       continue;
  //     }

  //     const int cend = particles_cells_end_gpu[node_hash];

  //     if (cend < 0) {
  //       continue;
  //     }

  //     for (int j = cstart; j < cend; j++) {
  //       const int particle_id = particles_sorted_indices_gpu[j];

  //       if (particle_is_rigid_gpu[particle_id]) {
  //         found_rigid = true;
  //       } else {
  //         found_non_rigid = true;
  //       }

  //       if (found_rigid && found_non_rigid) {
  //         // is_overlapping_gpu[node_mem_index] = true;
  //         is_overlapping_gpu[node_mem_index] = true;
  //         return;
  //       }
  //     }
  //   }
  //   is_overlapping_gpu[node_mem_index] = false;
  //   //   // bool is_particle_rigid = particle_is_rigid_gpu[tid];

  //   //   // if (!is_rigid_body && is_particle_rigid) {
  //   //   //   return;
  //   //   // }

  //   //   // if (is_rigid_body && !is_particle_rigid) {
  //   //   //   return;
  //   //   // }

  //   //   const Vectori particle_bin = particles_bins_gpu[tid];

  //   //   const Vectorr particle_coords = particles_positions_gpu[tid];

  //   // #ifdef CUDA_ENABLED
  //   //   const int num_surround_nodes = num_surround_nodes_gpu;
  //   // #else
  //   //   const int num_surround_nodes = num_surround_nodes_cpu;
  //   // #endif

  //   //   // We only search in a cube of 3x3x3 nodes around the particle
  //   //   const int linear_p2g_window_3d[64][3] = {
  //   //       {0, 0, 0},   {1, 0, 0},   {-1, 0, 0},  {0, 1, 0},   {1, 1, 0},
  //   //       {-1, 1, 0},  {0, -1, 0},  {1, -1, 0},  {-1, -1, 0}, {0, 0, 1},
  //   //       {1, 0, 1},   {-1, 0, 1},  {0, 1, 1},   {1, 1, 1},   {-1, 1, 1},
  //   //       {0, -1, 1},  {1, -1, 1},  {-1, -1, 1}, {0, 0, -1},  {1, 0, -1},
  //   //       {-1, 0, -1}, {0, 1, -1},  {1, 1, -1},  {-1, 1, -1}, {0, -1, -1},
  //   //       {1, -1, -1}, {-1, -1, -1}};

  //   //   // // #pragma unroll
  //   //   // for (int sid = 0; sid < 27; sid++) {

  //   // #pragma unroll
  //   //   for (int sid = 0; sid < num_surround_nodes; sid++) {

  //   //     // #ifdef CUDA_ENABLED
  //   //     //     const Vectori selected_bin = WINDOW_BIN(particle_bin,
  //   //     g2p_window_gpu,
  //   //     //     sid);
  //   //     // #else
  //   //     //     const Vectori selected_bin = WINDOW_BIN(particle_bin,
  //   //     g2p_window_gpu,
  //   //     //     sid);
  //   //     // #endif

  //   // #ifdef CUDA_ENABLED
  //   //     const Vectori selected_bin =
  //   //         WINDOW_BIN(particle_bin, linear_p2g_window_3d, sid);
  //   // #else
  //   //     const Vectori selected_bin =
  //   //         WINDOW_BIN(particle_bin, linear_p2g_window_3d, sid);
  //   // #endif
  //   //     const unsigned int node_hash = NODE_MEM_INDEX(selected_bin,
  //   //     grid.num_cells);

  //   //     if (node_hash >= grid.num_cells_total) {
  //   //       continue;
  //   //     }

  //   //     bool is_overlapping = true;

  //   //     const Vectorr relative_coordinates =
  //   //         (particle_coords - grid.origin) * grid.inv_cell_size -
  //   //         selected_bin.cast<Real>();

  //   //     // const Real radius = 1.;
  //   //     const Real radius = 1;
  //   //     if (fabs(relative_coordinates(0)) >= radius) {
  //   //       is_overlapping = false;
  //   //     }

  //   // #if DIM > 1
  //   //     if (fabs(relative_coordinates(1)) >= radius) {
  //   //       is_overlapping = false;
  //   //     }
  //   // #endif

  //   // #if DIM > 2
  //   //     if (fabs(relative_coordinates(2)) >= radius)
  //   //       is_overlapping = false;
  //   // #endif

  //   //     if (is_rigid_body) {
  //   //       is_overlapping_gpu[node_hash] = is_overlapping;
  //   //     } else {
  //   //       if (is_overlapping_gpu[node_hash]) {
  //   //         is_overlapping_gpu[node_hash] = is_overlapping;
  //   //       } else {
  //   //         is_overlapping_gpu[node_hash] = false;
  //   //       }
  //   //     }
  //   //     // printf("[%d] overlapping with node %d %d %d \n", tid,
  //   //     selected_bin[0],
  //   //     //  selected_bin[1], selected_bin[2]);
  //   //   }


}

#ifdef CUDA_ENABLED
__global__ void KERNEL_GET_OVERLAPPING_RIGID_BODY_GRID(
    bool *is_overlapping_gpu, const Vectorr *particles_positions_gpu,
    const Vectori *particles_bins_gpu, int *particles_cells_start_gpu,
    const int *particles_cells_end_gpu, const int *particles_sorted_indices_gpu,
    const bool *particle_is_rigid_gpu, const Grid grid,
    const int num_particles) {
  const int nid = blockDim.x * blockIdx.x + threadIdx.x;

  if (nid >= num_particles) {
    return;
  } // block access threads

  g2p_get_nodes_w_rigid_particles(
      is_overlapping_gpu, particles_positions_gpu, particles_bins_gpu,
      particles_cells_start_gpu, particles_cells_end_gpu,
      particles_sorted_indices_gpu, particle_is_rigid_gpu, grid, nid);
}

#endif
} // namespace pyroclastmpm