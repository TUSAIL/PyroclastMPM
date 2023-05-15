#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

__global__ void KERNELS_USL_P2G(Vectorr* nodes_moments_gpu,
                                Vectorr* nodes_forces_internal_gpu,
                                Real* nodes_masses_gpu,
                                const Vectori* node_ids_gpu,
                                const Matrix3r* particles_stresses_gpu,
                                const Vectorr *particles_forces_external_gpu,
                                const Vectorr* particles_velocities_gpu,
                                const Vectorr* particles_dpsi_gpu,
                                const Real* particles_psi_gpu,
                                const Real* particles_masses_gpu,
                                const Real* particles_volumes_gpu,
                                const int* particles_cells_start_gpu,
                                const int* particles_cells_end_gpu,
                                const int* particles_sorted_indices_gpu,
                                const Vectori num_nodes,
                                const Real inv_cell_size,
                                const int num_nodes_total) ;

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
                                 const Real alpha);

}  // namespace pyroclastmpm