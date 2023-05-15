#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm
{

    /**
     * @brief This is a kernel that calculates the bins and spatial hashes of
     * primitives. The grid size is usually the same as the nodal spacing.
     *
     * @param bins_gpu Output spatial bin of primitive
     * @param hashes_unsorted_gpu Output spatial hash of primitive
     * @param positions_gpu Position of primitive
     * @param grid_start Grid start origin
     * @param grid_end Grid end (extent)
     * @param num_cells Number of grid cells in each dimension
     * @param inv_cell_size Inverse cell size
     * @param num_elements Number of primitives
     */
    __global__ void KERNEL_CALC_HASH(Vectori *bins_gpu,
                                     unsigned int *hashes_unsorted_gpu,
                                     const Vectorr *positions_gpu,
                                     const Vectorr grid_start,
                                     const Vectorr grid_end,
                                     const Vectori num_cells,
                                     const Real inv_cell_size,
                                     const int num_elements);

    __global__ void KERNEL_BIN_PARTICLES(int *cell_start,
                                         int *cell_end,
                                         const unsigned int *hashes_sorted,
                                         const int num_elements);
} // namespace pyroclastmpm