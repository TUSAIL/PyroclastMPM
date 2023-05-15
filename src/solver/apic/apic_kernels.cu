#include "pyroclastmpm/solver/apic/apic_kernels.cuh"

namespace pyroclastmpm {

extern __constant__ Real dt_gpu;
extern __constant__ int dimension_global_gpu;
extern __constant__ int num_surround_nodes_gpu;
extern __constant__ int backward_window_gpu[64][3];
extern __constant__ int forward_window_gpu[64][3];

__global__ void KERNELS_USL_P2G_APIC(
    Vector3r* nodes_moments_gpu,
    Vector3r* nodes_forces_internal_gpu,
    Real* nodes_masses_gpu,
    const Vector3i* node_ids_3d_gpu,
    const Matrix3r* particles_stresses_gpu,
    const Matrix3r* particles_velocity_gradients_gpu,
    const Vector3r* particles_velocities_gpu,
    const Vector3r* particles_positions_gpu,
    const Vector3r* particles_dpsi_gpu,
    const Real* particles_psi_gpu,
    const Real* particles_masses_gpu,
    const Real* particles_volumes_gpu,
    const int* particles_cells_start_gpu,
    const int* particles_cells_end_gpu,
    const int* particles_sorted_indices_gpu,
    const Vector3i num_nodes,
    const Real inv_cell_size,
    const int num_nodes_total) {
  const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (node_mem_index >= num_nodes_total) {
    return;
  }
  Vector3i node_bin = node_ids_3d_gpu[node_mem_index];

  Vector3r total_node_moment = {0., 0., 0.};

  Vector3r total_node_force_internal = {0., 0., 0.};

  Real total_node_mass = 0.;

  for (int sid = 0; sid < num_surround_nodes_gpu; sid++) {
    const Vector3i backward_mesh = {backward_window_gpu[sid][0],
                                    backward_window_gpu[sid][1],
                                    backward_window_gpu[sid][2]};

    const Vector3i selected_bin = node_bin + backward_mesh;

    const Vector3r grid_pos = selected_bin.cast<Real>() / inv_cell_size;

    bool invalidCell = false;

    if (invalidCell) {
      continue;
    }

    const unsigned int node_hash =
        selected_bin[0] + selected_bin[1] * num_nodes[0] +
        selected_bin[2] * num_nodes[0] * num_nodes[1];

    // if (node_hash < 0) {
    //   continue;
    // }

    if (node_hash >= num_nodes_total) {
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

      const Real psi_particle =
          particles_psi_gpu[particle_id * num_surround_nodes_gpu + sid];

      const Vector3r dpsi_particle =
          particles_dpsi_gpu[particle_id * num_surround_nodes_gpu + sid];

      const Real scaled_mass = psi_particle * particles_masses_gpu[particle_id];

      total_node_mass += scaled_mass;

      // APIC
      const Vector3r rel_pos = grid_pos - particles_positions_gpu[particle_id];
      total_node_moment +=
          scaled_mass *
          (particles_velocities_gpu[particle_id] +
           particles_velocity_gradients_gpu[particle_id] * rel_pos);

      // END APIC

      total_node_force_internal += -1. * particles_volumes_gpu[particle_id] *
                                   particles_stresses_gpu[particle_id] *
                                   dpsi_particle;
    }
  }

  nodes_masses_gpu[node_mem_index] = total_node_mass;
  nodes_moments_gpu[node_mem_index] = total_node_moment;
  nodes_forces_internal_gpu[node_mem_index] = total_node_force_internal;
}

__global__ void KERNEL_USL_G2P_APIC(Matrix3r* particles_velocity_gradients_gpu,
                               Matrix3r* particles_F_gpu,
                               Matrix3r* particles_strains_gpu,
                               Matrix3r* particles_strain_increments,
                               Vector3r* particles_velocities_gpu,
                               Vector3r* particles_positions_gpu,
                               Real* particles_volumes_gpu,
                               Real* particles_densities_gpu,
                               const Vector3r* particles_dpsi_gpu,
                               const Vector3i* particles_bins_gpu,
                               const Real* particles_volumes_original_gpu,
                               const Real* particles_densities_original_gpu,
                               const Real* particles_psi_gpu,
                               const Real* particles_masses_gpu,
                               const Vector3r* nodes_moments_gpu,
                               const Vector3r* nodes_moments_nt_gpu,
                               const Real* nodes_masses_gpu,
                               const Matrix3r Wp_inverse,
                               const Real inv_cell_size,
                               const Vector3i num_cells,
                               const int num_particles) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  }  // block access threads

  const Vector3i particle_bin = particles_bins_gpu[tid];
  const Vector3r particle_coords = particles_positions_gpu[tid];

  Vector3r vel_inc = {0., 0., 0.};

  Vector3r dvel_inc = {0., 0., 0.};

  Matrix3r Bp = Matrix3r::Zero(); //APIC

  for (int i = 0; i < num_surround_nodes_gpu; i++) {
    const Vector3i forward_mesh = {forward_window_gpu[i][0],
                                   forward_window_gpu[i][1],
                                   forward_window_gpu[i][2]};

    const Vector3i selected_bin = particle_bin + forward_mesh;

    bool invalidCell = false;

    for (int axis = 0; axis < dimension_global_gpu; axis++) {
      if ((selected_bin[axis] < 0) || (selected_bin[axis] >= num_cells[axis])) {
        invalidCell = true;
        break;
      }
    }
    if (invalidCell) {
      continue;
    }

    const unsigned int nhash = selected_bin[0] +
                               selected_bin[1] * num_cells[0] +
                               selected_bin[2] * num_cells[0] * num_cells[1];

    const Real node_mass = nodes_masses_gpu[nhash];
    if (node_mass <= 0.000000001) {
      continue;
    }

    const Real psi_particle =
        particles_psi_gpu[tid * num_surround_nodes_gpu + i];
    const Vector3r dpsi_particle =
        particles_dpsi_gpu[tid * num_surround_nodes_gpu + i];

    const Vector3r node_velocity = nodes_moments_gpu[nhash] / node_mass;

    const Vector3r node_velocity_nt = nodes_moments_nt_gpu[nhash] / node_mass;

    const Vector3r delta_velocity = node_velocity_nt - node_velocity;

    dvel_inc += psi_particle * delta_velocity;

    vel_inc += psi_particle * node_velocity_nt;

    // APIC
    const Vector3r grid_pos = selected_bin.cast<Real>() / inv_cell_size;
    const Vector3r rel_pos = grid_pos - particle_coords;
    Bp += psi_particle*node_velocity_nt*rel_pos.transpose();
    // APIC get Bp
  }

  // Get velocity gradient APIC
  Matrix3r vel_grad = Bp*Wp_inverse;

  // END APIC

  // const Real alpha = 0.99;  // make this adjustable parameter

  Real alpha = 0; // Full PIC best?

  particles_velocities_gpu[tid] =
      alpha * (particles_velocities_gpu[tid] + dvel_inc) +
      (1 - alpha) * vel_inc;

  particles_positions_gpu[tid] = particle_coords + dt_gpu * vel_inc;

  particles_velocity_gradients_gpu[tid] = vel_grad;

  particles_F_gpu[tid] =
      (Matrix3r::Identity() + vel_grad * dt_gpu) * particles_F_gpu[tid];

  Real J = particles_F_gpu[tid].determinant();

  particles_volumes_gpu[tid] = J * particles_volumes_original_gpu[tid];

  particles_densities_gpu[tid] = particles_densities_original_gpu[tid] / J;

  Matrix3r deformation_matrix = 0.5 * (vel_grad + vel_grad.transpose());

  const Matrix3r strain_increments = deformation_matrix * dt_gpu;

  particles_strain_increments[tid] = strain_increments;

  particles_strains_gpu[tid] += strain_increments;
}

}  // namespace pyroclastmpm