#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm
{
    __global__ void KERNEL_CALCULATE_INITIAL_VOLUME(
        Real *particles_volumes_gpu,
        Real *particles_volumes_original_gpu,
        const Vectori *particles_bins_gpu,
        const int *particles_cells_start_gpu,
        const int *particles_cells_end_gpu,
        const Vectori num_cells,
        const Real cell_size,
        const int num_particles,
        const int num_nodes_total);

} // namespace pyroclastmpm