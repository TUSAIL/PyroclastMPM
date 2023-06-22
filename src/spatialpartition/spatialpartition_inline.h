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

/* @file spatialpartition_inline.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief CUDA kernels related to spatial partitioning
 * @version 0.1
 * @date 2023-06-17
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "pyroclastmpm/spatialpartition/spatialpartition.h"

namespace pyroclastmpm {

/// @brief Calculate the cartesian hash of a point based on the its position to
/// in the grid
/// @param bins_gpu spatial bin id (idx, idy, idz) of each point
/// @param hash_unsorted_gpu the points' spatial hashes
/// @param positions_gpu points' coordinates
/// @param grid information about the grid
/// @param tid id of the point
__device__ __host__ inline void
calculate_hashes(Vectori *bins_gpu, unsigned int *hash_unsorted_gpu,
                 const Vectorr *positions_gpu, const Grid &grid, const int tid)

{
  const Vectorr relative_position =
      (positions_gpu[tid] - grid.origin) * grid.inv_cell_size;

  bins_gpu[tid] = relative_position.cast<int>();

  hash_unsorted_gpu[tid] =
      NODE_MEM_INDEX(bins_gpu[tid],
                     grid.num_cells); // MACRO defined in type_commons.cuh
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_CALC_HASH(Vectori *bins_gpu,
                                 unsigned int *hashes_unsorted_gpu,
                                 const Vectorr *positions_gpu, const Grid grid,
                                 const int num_elements) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_elements) {
    return;
  } // block access threads

  calculate_hashes(bins_gpu, hashes_unsorted_gpu, positions_gpu, grid, tid);
}
#endif

/// @brief Populate the cell start and end range of each point
/// @param cell_start_gpu cell start range of each point
/// @param cell_end_gpu cell end range of each point
/// @param hash_sorted sorted spatial hashes of each point
/// @param num_elements number of points
/// @param tid id of the point
__host__ __device__ inline void
bin_particles_kernel(int *cell_start_gpu, int *cell_end_gpu,
                     const unsigned int *hash_sorted, const int num_elements,
                     const int tid) {

  if (tid >= num_elements) {
    return;
  } // block access threads

  unsigned int hash;
  unsigned int nexthash;
  hash = hash_sorted[tid];

  if (tid < num_elements - 1) {
    nexthash = hash_sorted[tid + 1];

    if (tid == 0) {
      cell_start_gpu[hash] = tid;
    }

    if (hash != nexthash) {
      cell_end_gpu[hash] = tid + 1;

      cell_start_gpu[nexthash] = tid + 1;
    }
  }

  if (tid == num_elements - 1) {
    cell_end_gpu[hash] = tid + 1;
  }
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_BIN_PARTICLES(int *cell_start_gpu, int *cell_end_gpu,
                                     const unsigned int *hash_sorted,
                                     const int num_elements) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  bin_particles_kernel(cell_start_gpu, cell_end_gpu, hash_sorted, num_elements,
                       tid);
}
#endif

} // namespace pyroclastmpm