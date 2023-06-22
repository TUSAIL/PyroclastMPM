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

/**
 * @file spatialpartition.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Implementation of the SpatialPartition class
 * @details calls the calculate_hash, sort_hash and bin_particles functions
 * to bin and partition the particles into a grid.
 *
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#include "pyroclastmpm/spatialpartition/spatialpartition.h"

#include "spatialpartition_inline.h"

namespace pyroclastmpm {

/// @brief Construct a new Spatial Partition object
/// @param _grid Grid object (@see Grid)
/// @param _num_elements number of particles (or points) being partitioned
SpatialPartition::SpatialPartition(const Grid &_grid, const int _num_elements)
    : grid(_grid), num_elements(_num_elements) {

  set_default_device<int>(grid.num_cells_total, {}, cell_start_gpu, -1);
  set_default_device<int>(grid.num_cells_total, {}, cell_end_gpu, -1);
  set_default_device<int>(num_elements, {}, sorted_index_gpu, -1);
  set_default_device<unsigned int>(num_elements, {}, hash_unsorted_gpu, 0);
  set_default_device<unsigned int>(num_elements, {}, hash_sorted_gpu, 0);
  set_default_device<Vectori>(num_elements, {}, bins_gpu, Vectori::Zero());

  reset();
#ifdef CUDA_ENABLED
  launch_config = GPULaunchConfig(num_elements);
#endif
}

/// @brief Resets arrays of the Spatial Partition object
void SpatialPartition::reset() {

  thrust::sequence(sorted_index_gpu.begin(), sorted_index_gpu.end(), 0, 1);
  thrust::fill(cell_start_gpu.begin(), cell_start_gpu.end(), -1);
  thrust::fill(cell_end_gpu.begin(), cell_end_gpu.end(), -1);
  thrust::fill(hash_sorted_gpu.begin(), hash_sorted_gpu.end(), 0);
  thrust::fill(hash_unsorted_gpu.begin(), hash_unsorted_gpu.end(), 0);
  thrust::fill(bins_gpu.begin(), bins_gpu.end(), Vectori::Zero());
}

/// @brief Calculates the cartesian hash of a set of points
/// @param positions_gpu Set of points with the same size as _num_elements
void SpatialPartition::calculate_hash(gpu_array<Vectorr> &positions_gpu) {

#ifdef CUDA_ENABLED
  KERNEL_CALC_HASH<<<launch_config.tpb, launch_config.bpg>>>(
      thrust::raw_pointer_cast(bins_gpu.data()),
      thrust::raw_pointer_cast(hash_unsorted_gpu.data()),
      thrust::raw_pointer_cast(positions_gpu.data()), grid, num_elements);
  gpuErrchk(cudaDeviceSynchronize());
#else

  for (int index = 0; index < num_elements; index++) {
    calculate_hashes(bins_gpu.data(), hash_unsorted_gpu.data(),
                     positions_gpu.data(), grid, index);
  }

#endif

  hash_sorted_gpu = hash_unsorted_gpu;
}

/// @brief Sorts hashes of the points
void SpatialPartition::sort_hashes() {
  thrust::stable_sort_by_key(hash_sorted_gpu.begin(), hash_sorted_gpu.end(),
                             sorted_index_gpu.begin());
}

/// @brief Bins the points into the grid
/// @details Calculates the start and end cell index of each point
void SpatialPartition::bin_particles() {
#ifdef CUDA_ENABLED
  KERNEL_BIN_PARTICLES<<<launch_config.tpb, launch_config.bpg>>>(
      thrust::raw_pointer_cast(cell_start_gpu.data()),
      thrust::raw_pointer_cast(cell_end_gpu.data()),
      thrust::raw_pointer_cast(hash_sorted_gpu.data()), num_elements);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (size_t ti = 0; ti < num_elements; ti++) {
    bin_particles_kernel(cell_start_gpu.data(), cell_end_gpu.data(),
                         hash_sorted_gpu.data(), num_elements, ti);
  }
#endif
}

} // namespace pyroclastmpm