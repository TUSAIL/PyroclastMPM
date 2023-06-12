// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
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

#pragma once

#include "pyroclastmpm/common/helper.h"
#include "pyroclastmpm/common/types_common.h"
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pyroclastmpm {

/*!
 * @brief Spatial partitioning class
 */
class SpatialPartition {
public:
  /**
   * @brief Construct a new Spatial Partition object.
   * @param _node_start start of the spatial partitioning domain
   * @param _node_end end of the spatial partitioning domain
   * @param _node_spacing cell size of the domain
   * @param _num_elements number of particles (or stl centroid being
   * partitioned) into the grid
   */
  SpatialPartition(const Vectorr _node_start, const Vectorr _node_end,
                   const Real _node_spacing, const int _num_elements);

  /**
   * @brief Default constructor to create a temporary
   *
   */
  SpatialPartition() = default;

  /** @brief Destroy the Spatial Partition object */
  ~SpatialPartition();

  /** @brief Resets the memory of the Spatial Partition object */
  void reset();

  /**
   * @brief Calculates the cartesian hash of a set of coordinates.
   * @param positions_gpu Set of coordinates with the same size as _num_elements
   */
  void calculate_hash(gpu_array<Vectorr> &positions_gpu);

  /** @brief Sort hashes and keys */
  void sort_hashes();

  /** @brief Bin incies in the cells */
  void bin_particles();

  /** @brief start indices of the bins */
  gpu_array<int> cell_start_gpu;

  /** @brief end indices of the bins */
  gpu_array<int> cell_end_gpu;

  /** @brief sorted indices of the coordinates */
  gpu_array<int> sorted_index_gpu;

  /** @brief unsorted cartesian hashes of the coordinates */
  gpu_array<unsigned int> hash_unsorted_gpu;

  /** @brief sorted cartesian hashes with respect to the sorted indices. */
  gpu_array<unsigned int> hash_sorted_gpu;

  /** @brief the bins (counts) of particles/elements within a cell */
  gpu_array<Vectori> bins_gpu;

  /** @brief start domain of the partitioning grid */
  Vectorr grid_start;

  /** @brief end domain of the partitioning grid */
  Vectorr grid_end;

  /** @brief cell size of the partitioning grid */
  Real cell_size;

  /** @brief inverse cell size of the partitioning grid */
  Real inv_cell_size;

  /** @brief number of cells */
  Vectori num_cells;

  /** @brief number of elements being partitioned*/
  int num_elements;

  /** @brief total number of cells within the grid */
  int num_cells_total;

#ifdef CUDA_ENABLED
  GPULaunchConfig launch_config;
#endif
};
} // namespace pyroclastmpm