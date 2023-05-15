#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

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
    const int num_nodes_total);

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
                               const int num_particles);

}  // namespace pyroclastmpm