#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm
{

    __global__ void KERNELS_CALC_NON_RIGID_GRID_NORMALS(
        Vectorr *grid_normals_gpu,
        const Vectori *node_ids_gpu,
        const Vectorr *particles_dpsi_gpu,
        const Real *particles_masses_gpu,
        const int *particles_cells_start_gpu,
        const int *particles_cells_end_gpu,
        const int *particles_sorted_indices_gpu,
        const Vectori num_nodes,
        const int num_nodes_total);

    __global__ void KERNEL_GET_OVERLAPPING_RIGID_BODY_GRID(
        bool *is_overlapping_gpu,
        const Vectori *node_ids_gpu,
        const Vectorr *particles_positions_gpu,
        const Vectori *particles_bins_gpu,
        const Vectori num_nodes,
        const Vectorr origin,
        const Real inv_cell_size,
        const int num_nodes_total,
        const int num_particles);

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
                                              const int num_nodes_total);

    __global__ void KERNEL_UPDATE_POS_RIGID(Vectorr *particles_positions_gpu,
                                            Vectorr *particles_velocities_gpu,
                                            const Matrixr rotation_matrix,
                                            const Vectorr COM,
                                            const Vectorr translational_velocity,
                                            const int num_particles);

    __global__ void KERNEL_FIND_NEAREST_RIGIDPARTICLE(
        int* closest_rigid_id_gpu,
        const Vectorr* rigid_positions_gpu,
        const Vectori* node_ids_gpu,
        const Real* nodes_masses_gpu,
        const int* rigid_cells_start_gpu,
        const int* rigid_cells_end_gpu,
        const int* rigid_sorted_indices_gpu,
        const bool* is_overlapping_gpu,
        const Vectori num_nodes,
        const Vectorr origin,
        const Real inv_cell_size,
        const int num_nodes_total);

} // namespace pyroclastmpm