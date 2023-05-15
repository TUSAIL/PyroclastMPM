#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

// __global__ void KERNELS_P2G_RIGID(Vector3r* nodes_moments_nt_gpu,
//                                   Vector3r* nodes_moments_gpu,
//                                   Real* nodes_masses_gpu,
//                                   const Vector3i* node_ids_3d_gpu,
//                                   const Vector3r* particles_positions_gpu,
//                                   const Vector3r* particles_velocities_gpu,
//                                   const Real* particles_psi_gpu,
//                                   const int* particles_cells_start_gpu,
//                                   const int* particles_cells_end_gpu,
//                                   const int* particles_sorted_indices_gpu,
//                                   const Vector3i num_nodes,
//                                   const Real inv_cell_size,
//                                   const int num_nodes_total);

__global__ void KERNELS_CALC_GRID_NORMALS(
    Vector3r* grid_normals_gpu,
    const Real* nodes_masses_gpu,
    const Vector3i* node_ids_3d_gpu,
    const Vector3r* particles_dpsi_gpu,
    const Real* particles_masses_gpu,
    const int* particles_cells_start_gpu,
    const int* particles_cells_end_gpu,
    const int* particles_sorted_indices_gpu,
    const Vector3i num_nodes,
    const int num_nodes_total);

__global__ void KERNEL_GET_OVERLAPPING(bool* is_overlapping_gpu,
                                       const Vector3r* grid_normals_gpu,
                                       const Real* nodes_masses_gpu,
                                       const Vector3i* node_ids_3d_gpu,
                                       const Vector3r* particles_positions_gpu,
                                       const Vector3i* particles_bins_gpu,
                                       const Vector3i num_nodes,
                                       const Vector3r origin,
                                       const Real inv_cell_size,
                                       const int num_nodes_total,
                                       const int num_particles);

__global__ void KERNEL_VELOCITY_CORRECTOR(Vector3r* nodes_moments_nt_gpu,
                                          Vector3r* nodes_moments_gpu,
                                          const Real* nodes_masses_gpu,
                                          const Vector3r* grid_normals_gpu,
                                          bool* is_overlapping_gpu,
                                          const Vector3r body_vel,
                                          const int num_nodes_total);

__global__ void KERNEL_UPDATE_POS_RIGID(Vector3r* particles_positions_gpu,
                                        Vector3r* particles_velocities_gpu,
                                        const int num_particles);

}  // namespace pyroclastmpm