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

#include "pyroclastmpm/spatialpartition/spatialpartition.h"

namespace pyroclastmpm {

#include "spatialpartition_inline.h"

/**
 * @brief Construct a new Spatial Partition:: Spatial Partition object
 *
 * @param _node_start
 * @param _node_end
 * @param _node_spacing
 * @param _num_elements
 */
SpatialPartition::SpatialPartition(const Vectorr _node_start,
                                   const Vectorr _node_end,
                                   const Real _node_spacing,
                                   const int _num_elements)
    : grid_start(_node_start), grid_end(_node_end), cell_size(_node_spacing),
      num_elements(_num_elements)

{
  inv_cell_size = 1. / cell_size;
  num_cells_total = 1;
  num_cells = Vectori::Ones();

  for (int axis = 0; axis < DIM; axis++) {
    num_cells[axis] =
        (int)((grid_end[axis] - grid_start[axis]) / cell_size) + 1;
    num_cells_total *= num_cells[axis];
  }

  set_default_device<int>(num_cells_total, {}, cell_start_gpu, -1);
  set_default_device<int>(num_cells_total, {}, cell_end_gpu, -1);
  set_default_device<int>(num_elements, {}, sorted_index_gpu, -1);
  set_default_device<unsigned int>(num_elements, {}, hash_unsorted_gpu, 0);
  set_default_device<unsigned int>(num_elements, {}, hash_sorted_gpu, 0);
  set_default_device<Vectori>(num_elements, {}, bins_gpu, Vectori::Zero());
  reset();

#ifdef CUDA_ENABLED
  launch_config.tpb = dim3(int((num_cells_total) / BLOCKSIZE) + 1, 1, 1);
  launch_config.bpg = dim3(BLOCKSIZE, 1, 1);
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

/**
 * @brief Destroy the Spatial Partition:: Spatial Partition object
 *
 */
SpatialPartition::~SpatialPartition() {}

/**
 * @brief Reset the spatial partition arrays
 *
 */
void SpatialPartition::reset() {
  thrust::sequence(sorted_index_gpu.begin(), sorted_index_gpu.end(), 0, 1);
  thrust::fill(cell_start_gpu.begin(), cell_start_gpu.end(), -1);
  thrust::fill(cell_end_gpu.begin(), cell_end_gpu.end(), -1);
  thrust::fill(hash_sorted_gpu.begin(), hash_sorted_gpu.end(), 0);
  thrust::fill(hash_unsorted_gpu.begin(), hash_unsorted_gpu.end(), 0);
  thrust::fill(bins_gpu.begin(), bins_gpu.end(), Vectori::Zero());
}

/**
 * @brief Sorts the particles by their hash value
 *
 * @param positions_gpu
 */
void SpatialPartition::calculate_hash(gpu_array<Vectorr> &positions_gpu) {
#ifdef CUDA_ENABLED
  KERNEL_CALC_HASH<<<launch_config.tpb, launch_config.bpg>>>(
      thrust::raw_pointer_cast(bins_gpu.data()),
      thrust::raw_pointer_cast(hash_unsorted_gpu.data()),
      thrust::raw_pointer_cast(positions_gpu.data()), grid_start, grid_end,
      num_cells, inv_cell_size, num_elements);
  gpuErrchk(cudaDeviceSynchronize());
#else

  for (int hi = 0; hi < num_elements; hi++) {
    /* code */
    calculate_hashes(bins_gpu.data(), hash_unsorted_gpu.data(),
                     positions_gpu.data(), grid_start, grid_end, num_cells,
                     inv_cell_size, hi);
  }

#endif

  hash_sorted_gpu = hash_unsorted_gpu; // move this inside kernel?
}

/**
 * @brief Sorts the particles by their hash value
 *
 */
void SpatialPartition::sort_hashes() {
  thrust::stable_sort_by_key(hash_sorted_gpu.begin(), hash_sorted_gpu.end(),
                             sorted_index_gpu.begin());
}

/**
 * @brief calculates the start and end of each cell in the grid containing the
 * particles
 *
 */
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