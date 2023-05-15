#include "pyroclastmpm/solver/usl/usl_kernels.cuh"

namespace pyroclastmpm
{

  extern __constant__ Real dt_gpu;
  extern __constant__ int dimension_global_gpu;
  extern __constant__ int num_surround_nodes_gpu;
  extern __constant__ int backward_window_gpu[64][3];
  extern __constant__ int forward_window_gpu[64][3];

  /**
   * @brief Computes the velocity gradient for a given particle (velocities
   * gathered from surrounding nodes).
   *
   * @param nodes_moments_gpu
   * @param nodes_forces_internal_gpu
   * @param nodes_masses_gpu
   * @param node_ids_3d_gpu
   * @param particles_stresses_gpu
   * @param particles_velocities_gpu
   * @param particles_dpsi_gpu
   * @param particles_psi_gpu
   * @param particles_masses_gpu
   * @param particles_volumes_gpu
   * @param particles_cells_start_gpu
   * @param particles_cells_end_gpu
   * @param particles_sorted_indices_gpu
   * @param num_nodes
   * @param inv_cell_size
   * @param num_nodes_total
   */
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

    const Vectori node_bin = node_ids_gpu[node_mem_index];

    Vectorr total_node_moment = Vectorr::Zero();

    Vectorr total_node_force_internal = Vectorr::Zero();

    Vectorr total_node_force_external = Vectorr::Zero();

    Real total_node_mass = 0.;

#pragma unroll
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

        const Real psi_particle =
            particles_psi_gpu[particle_id * num_surround_nodes_gpu + sid];

        const Vectorr dpsi_particle =
            particles_dpsi_gpu[particle_id * num_surround_nodes_gpu + sid];

        const Real scaled_mass = psi_particle * particles_masses_gpu[particle_id];

        total_node_mass += scaled_mass;

        total_node_moment += scaled_mass * particles_velocities_gpu[particle_id];

        total_node_force_external += psi_particle*particles_forces_external_gpu[particle_id];
#if DIM == 3
        total_node_force_internal += -1. * particles_volumes_gpu[particle_id] *
                                     particles_stresses_gpu[particle_id] *
                                     dpsi_particle;
#else
        Matrix3r cauchy_stress_3d = particles_stresses_gpu[particle_id];
        Matrixr cauchy_stress = cauchy_stress_3d.block(0, 0, DIM, DIM);
        total_node_force_internal += -1. * particles_volumes_gpu[particle_id] *
                                     cauchy_stress *
                                     dpsi_particle;

#endif
      }
    }

    nodes_masses_gpu[node_mem_index] = total_node_mass;
    nodes_moments_gpu[node_mem_index] = total_node_moment;
    nodes_forces_internal_gpu[node_mem_index] = total_node_force_internal + total_node_force_external;
  }

  /**
   * @brief Grid to Particle (G2P) kernel for USL solver (gather velocities from particles to nodes)
   *
   * @param particles_velocity_gradients_gpu velocity gradient of particles
   * @param particles_F_gpu Deformation gradient of particles
   * @param particles_velocities_gpu particle velocities
   * @param particles_positions_gpu particle positions
   * @param particles_volumes_gpu particle volumes
   * @param particles_dpsi_gpu gradient of the shape functions
   * @param particles_bins_gpu particle bins (spatial hash location)
   * @param particles_volumes_original_gpu original particle volumes
   * @param particles_psi_gpu particle/node shape functions
   * @param particles_masses_gpu particle masses
   * @param nodes_moments_gpu node moments (velocity * mass)
   * @param nodes_moments_nt_gpu node moments predictor
   * @param nodes_masses_gpu node masses
   * @param num_cells number of cells in the grid (spatial bin)
   * @param num_particles
   */
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
                                 const Real alpha
                                 )
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_particles)
    {
      return;
    } // block access threads

    const Vectori particle_bin = particles_bins_gpu[tid];
    const Vectorr particle_coords = particles_positions_gpu[tid];

    Vectorr vel_inc = Vectorr::Zero();
    Vectorr dvel_inc = Vectorr::Zero();
    Matrixr vel_grad = Matrixr::Zero();

    for (int i = 0; i < num_surround_nodes_gpu; i++)
    {
#if DIM == 3
      const Vectori forward_mesh = Vectori({forward_window_gpu[i][0],
                                            forward_window_gpu[i][1],
                                            forward_window_gpu[i][2]});

      const Vectori selected_bin = particle_bin + forward_mesh;

      const unsigned int nhash = selected_bin[0] +
                                 selected_bin[1] * num_cells[0] +
                                 selected_bin[2] * num_cells[0] * num_cells[1];

#elif DIM == 2
      const Vectori forward_mesh = Vectori({forward_window_gpu[i][0],
                                            forward_window_gpu[i][1]});

      const Vectori selected_bin = particle_bin + forward_mesh;

      const unsigned int nhash = selected_bin[0] +
                                 selected_bin[1] * num_cells[0];
#else
      const Vectori forward_mesh = Vectori(forward_window_gpu[i][0]);

      const Vectori selected_bin = particle_bin + forward_mesh;

      const unsigned int nhash = selected_bin[0];
#endif

      bool invalidCell = false;

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

      const Real psi_particle =
          particles_psi_gpu[tid * num_surround_nodes_gpu + i];
      const Vectorr dpsi_particle =
          particles_dpsi_gpu[tid * num_surround_nodes_gpu + i];

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

    particles_positions_gpu[tid] = particle_coords + dt_gpu * vel_inc;

    particles_F_gpu[tid] =
        (Matrixr::Identity() + vel_grad * dt_gpu) * particles_F_gpu[tid];

    Real J = particles_F_gpu[tid].determinant();

    particles_volumes_gpu[tid] = J * particles_volumes_original_gpu[tid];
  }

} // namespace pyroclastmpm