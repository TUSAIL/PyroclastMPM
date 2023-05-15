#include "../common/types_common.cuh"

namespace pyroclastmpm {

/**
 * @brief
 *
 * @param particles_dpsi_gpu Output gradient of the shape function
 * @param particles_psi_gpu Output shape function
 * @param particles_positions_gpu Particle positions
 * @param particles_bins_gpu Particle bins found by the partitioning algorithm
 * @param num_particles Total number of particles
 * @param num_surround_nodes Number of surrounding nodes per node (8 for linear
 * and 64 for quadratic)
 * @param num_cells Number of cells (relative to partitioning algorithm)
 * @param inv_cell_size Inverse cell size (relative to partitioning algorithm)
 * @param shape_function The selected shape function
 */
__global__ void KERNEL_CALC_SHP(Vectorr* particles_dpsi_gpu,
                                Real* particles_psi_gpu,
                                const Vectorr* particles_positions_gpu,
                                const Vectori* particles_bins_gpu,
                                const Vectori* node_types_gpu,
                                const Vectori num_cells,
                                const Vectorr origin,
                                const Real inv_cell_size,
                                const int num_particles,
                                const int num_nodes_total);

                                

}