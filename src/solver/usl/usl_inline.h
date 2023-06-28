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

/**
 * @file usl_inline.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Kernels for Update Stress last solver
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern __constant__ Real dt_gpu;
extern __constant__ int num_surround_nodes_gpu;
extern __constant__ int g2p_window_gpu[64][3];
extern __constant__ int p2g_window_gpu[64][3];
#else
extern const Real dt_cpu;
extern const int num_surround_nodes_cpu;
extern const int g2p_window_cpu[64][3];
extern const int p2g_window_cpu[64][3];
#endif

/**
 * @brief Particle to Grid Transfer step
 * @details This function loops over the grid nodes then uses a connectivity
 * array to gather the velocities of the particles in neighboring cells
 * (depending on the order of the shape function).
 *
 * At the end of the kernel the nodal masses, moments and internal forces are
 * calculated
 *
 * @param nodes_moments_gpu node moments
 * @param nodes_forces_internal_gpu node internal force vector
 * @param nodes_masses_gpu node scalar (lump) masses
 * @param node_ids_gpu node bin ids (idx,idy,idz)
 * @param particles_stresses_gpu stress field of the particles
 * @param particles_forces_external_gpu external forces of the particles
 * @param particles_velocities_gpu velocity field of the particles
 * @param particles_dpsi_gpu node/particles gradient shape functions
 * @param particles_psi_gpu node/particles shape functions
 * @param particles_masses_gpu scalar masses of the particles
 * @param particles_volumes_gpu (updated) volumes of the particles
 * @param particles_cells_start_gpu start cell index at which a particle is
 * binned
 * @param particles_cells_end_gpu end cell inde at which a particle is binned
 * @param particles_sorted_indices_gpu sorted index according to their cartesian
 * hash
 * @param particles_is_rigid_gpu flag to indicate if particle is rigid
 * @param particles_is_active_gpu flag to indicate if a particle is active
 * @param grid background grid information
 * @param node_mem_index memory index of the particles
 */
__device__ __host__ void inline usl_p2g_kernel(
    Vectorr *nodes_moments_gpu, Vectorr *nodes_forces_internal_gpu,
    Real *nodes_masses_gpu, const Vectori *node_ids_gpu,
    const Matrix3r *particles_stresses_gpu,
    const Vectorr *particles_forces_external_gpu,
    const Vectorr *particles_velocities_gpu, const Vectorr *particles_dpsi_gpu,
    const Real *particles_psi_gpu, const Real *particles_masses_gpu,
    const Real *particles_volumes_gpu, const int *particles_cells_start_gpu,
    const int *particles_cells_end_gpu, const int *particles_sorted_indices_gpu,
    const bool *particles_is_rigid_gpu, const bool *particles_is_active_gpu,
    const Grid &grid, const int node_mem_index) {

  // loops over grid nodes

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
  for (int sid = 0; sid < num_surround_nodes; sid++) {
#ifdef CUDA_ENABLED
    const Vectori selected_bin = WINDOW_BIN(node_bin, p2g_window_gpu, sid);
#else
    const Vectori selected_bin = WINDOW_BIN(node_bin, p2g_window_cpu, sid);
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

      if (particles_is_rigid_gpu[particle_id] ||
          !particles_is_active_gpu[particle_id]) {
        continue;
      }

      const Real psi_particle =
          particles_psi_gpu[particle_id * num_surround_nodes + sid];

      Vector3r dpsi_particle = Vector3r::Zero();
      dpsi_particle.block(0, 0, DIM, DIM) =
          particles_dpsi_gpu[particle_id * num_surround_nodes + sid];

      const Real scaled_mass = psi_particle * particles_masses_gpu[particle_id];
      total_node_mass += scaled_mass;
      total_node_moment += scaled_mass * particles_velocities_gpu[particle_id];
      total_node_force_external +=
          psi_particle * particles_forces_external_gpu[particle_id];
      total_node_force_internal.block(0, 0, DIM, DIM) +=
          -1. * particles_volumes_gpu[particle_id] *
          particles_stresses_gpu[particle_id] * dpsi_particle;
    }
  }

  nodes_masses_gpu[node_mem_index] = total_node_mass;
  nodes_moments_gpu[node_mem_index] = total_node_moment;
  nodes_forces_internal_gpu[node_mem_index] =
      total_node_force_internal + total_node_force_external;
}

#ifdef CUDA_ENABLED
__global__ void KERNELS_USL_P2G(
    Vectorr *nodes_moments_gpu, Vectorr *nodes_forces_internal_gpu,
    Real *nodes_masses_gpu, const Vectori *node_ids_gpu,
    const Matrix3r *particles_stresses_gpu,
    const Vectorr *particles_forces_external_gpu,
    const Vectorr *particles_velocities_gpu, const Vectorr *particles_dpsi_gpu,
    const Real *particles_psi_gpu, const Real *particles_masses_gpu,
    const Real *particles_volumes_gpu, const int *particles_cells_start_gpu,
    const int *particles_cells_end_gpu, const int *particles_sorted_indices_gpu,
    const bool *particles_is_rigid_gpu, const bool *particles_is_active_gpu,
    const Grid grid) {

  const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (node_mem_index >= grid.num_cells_total) {
    return;
  }

  usl_p2g_kernel(
      nodes_moments_gpu, nodes_forces_internal_gpu, nodes_masses_gpu,
      node_ids_gpu, particles_stresses_gpu, particles_forces_external_gpu,
      particles_velocities_gpu, particles_dpsi_gpu, particles_psi_gpu,
      particles_masses_gpu, particles_volumes_gpu, particles_cells_start_gpu,
      particles_cells_end_gpu, particles_sorted_indices_gpu,
      particles_is_rigid_gpu, particles_is_active_gpu, grid, node_mem_index);
}

#endif

/**
 * @brief Grid to particle update (velocity scatter)
 * @details This function loops over each particle and scatters their velocity
 * to neighboring nodes (depending on the range of their shape function). It
 * uses a connectivity matrix to know which particles are receiving the fields.
 *
 * At the end of this functions the particles' velocities, velocity gradient,
 * update velume, deformation gradient and positions are calculated
 *
 * @param particles_velocity_gradients_gpu Velocity gradient of the particles
 * @param particles_F_gpu Deformation gradient of the particles
 * @param particles_velocities_gpu Velocities of the particles
 * @param particles_positions_gpu Positions of the particles
 * @param particles_volumes_gpu (Updated) velocities of the particles
 * @param particles_dpsi_gpu Node/particle gradient of the shape functions
 * @param particles_bins_gpu The bin (idx,idy,idz) of a particle on the
 * background grid
 * @param particles_volumes_original_gpu The (original) volume of the particle
 * @param particles_psi_gpu Node/particle shape function
 * @param particles_is_rigid_gpu flag if particle is rigid or not
 * @param particles_is_active_gpu flag if particle is active
 * @param nodes_moments_gpu Nodal moments
 * @param nodes_moments_nt_gpu Forward nodal moments (refer to paper)
 * @param nodes_masses_gpu nodal masses
 * @param grid information on the background grid
 * @param alpha FLIP/PIC ratio
 * @param tid Id of the particle
 * @return __device__
 */
__device__ __host__ inline void usl_g2p_kernel(
    Matrixr *particles_velocity_gradients_gpu, Matrixr *particles_F_gpu,
    Vectorr *particles_velocities_gpu, Vectorr *particles_positions_gpu,
    Real *particles_volumes_gpu, const Vectorr *particles_dpsi_gpu,
    const Vectori *particles_bins_gpu,
    const Real *particles_volumes_original_gpu, const Real *particles_psi_gpu,
    const bool *particles_is_rigid_gpu, const bool *particles_is_active_gpu,
    const Vectorr *nodes_moments_gpu, const Vectorr *nodes_moments_nt_gpu,
    const Real *nodes_masses_gpu, const Grid &grid, const Real alpha,
    const int tid) {

  if (particles_is_rigid_gpu[tid] || !particles_is_active_gpu[tid]) {
    return;
  }
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

  for (int i = 0; i < num_surround_nodes; i++) {

#ifdef CUDA_ENABLED
    const Vectori selected_bin = WINDOW_BIN(particle_bin, g2p_window_gpu, i);
#else
    const Vectori selected_bin = WINDOW_BIN(particle_bin, g2p_window_cpu, i);
#endif
    bool invalidCell = false;
    const unsigned int nhash = NODE_MEM_INDEX(selected_bin, grid.num_cells);
    // TODO this is slow!
    for (int axis = 0; axis < DIM; axis++) {
      if ((selected_bin[axis] < 0) ||
          (selected_bin[axis] >= grid.num_cells[axis])) {
        invalidCell = true;
        break;
      }
    }
    if (invalidCell) {
      continue;
    }

    const Real node_mass = nodes_masses_gpu[nhash];
    if (node_mass <= 0.000000001) {
      continue;
    }

    const Real psi_particle = particles_psi_gpu[tid * num_surround_nodes + i];
    const Vectorr dpsi_particle =
        particles_dpsi_gpu[tid * num_surround_nodes + i];

    const Vectorr node_velocity = nodes_moments_gpu[nhash] / node_mass;
    const Vectorr node_velocity_nt = nodes_moments_nt_gpu[nhash] / node_mass;
    const Vectorr delta_velocity = node_velocity_nt - node_velocity;

    dvel_inc += psi_particle * delta_velocity;
    vel_inc += psi_particle * node_velocity_nt;
    vel_grad += dpsi_particle * node_velocity_nt.transpose();
  }
  particles_velocities_gpu[tid] =
      alpha * (particles_velocities_gpu[tid] + dvel_inc) +
      (1. - alpha) * vel_inc;
  particles_velocity_gradients_gpu[tid] = vel_grad;
#ifdef CUDA_ENABLED
  particles_positions_gpu[tid] = particle_coords + dt_gpu * vel_inc;
  particles_F_gpu[tid] =
      (Matrixr::Identity() + vel_grad * dt_gpu) * particles_F_gpu[tid];
#else
  particles_positions_gpu[tid] = particle_coords + dt_cpu * vel_inc;
  particles_F_gpu[tid] =
      (Matrixr::Identity() + vel_grad * dt_cpu) * particles_F_gpu[tid];
#endif
  Real J = particles_F_gpu[tid].determinant();
  particles_volumes_gpu[tid] = J * particles_volumes_original_gpu[tid];
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_USL_G2P(
    Matrixr *particles_velocity_gradients_gpu, Matrixr *particles_F_gpu,
    Vectorr *particles_velocities_gpu, Vectorr *particles_positions_gpu,
    Real *particles_volumes_gpu, const Vectorr *particles_dpsi_gpu,
    const Vectori *particles_bins_gpu,
    const Real *particles_volumes_original_gpu, const Real *particles_psi_gpu,
    const bool *particles_is_rigid_gpu, const bool *particles_is_active_gpu,
    const Vectorr *nodes_moments_gpu, const Vectorr *nodes_moments_nt_gpu,
    const Real *nodes_masses_gpu, const Grid grid, const Real alpha,
    const int num_particles) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  usl_g2p_kernel(particles_velocity_gradients_gpu, particles_F_gpu,
                 particles_velocities_gpu, particles_positions_gpu,
                 particles_volumes_gpu, particles_dpsi_gpu, particles_bins_gpu,
                 particles_volumes_original_gpu, particles_psi_gpu,
                 particles_is_rigid_gpu, particles_is_active_gpu,
                 nodes_moments_gpu, nodes_moments_nt_gpu, nodes_masses_gpu,
                 grid, alpha, tid);
}
#endif

} // namespace pyroclastmpm