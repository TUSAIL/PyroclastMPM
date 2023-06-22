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

/**
 * @file spatialpartition.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Declaration of the SpatialPartition class
 * @details SpatialPartition class is used to bin
 * and partition the particles into a grid. This
 *  helps efficient access to the P2G and G2P kernels
 * (amongst others).
 *
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include "pyroclastmpm/common/helper.h"
#include "pyroclastmpm/common/types_common.h"
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pyroclastmpm {

/**
 * @brief Spatially partitions points into a grid
 * @details This is a "neighborhood search" algorithm that bins points into a
 * uniform grid which helps to efficiently access to the P2G and G2P kernels.
 *
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/nodes/nodes.h"
 *
 *        SpatialPartition spatial =
 *            SpatialPartition(Grid(grid_origin, grid_end, cell_size),
 * num_points);
 *
 *        spatial.reset();
 *
 *        spatial.calculate_hash(point_coordinates);
 *
 *        spatial.sort_hashes();
 *
 *        spatial.bin_particles();
 *
 * \endverbatim
 */
class SpatialPartition {
public:
  /// @brief Construct a new Spatial Partition object
  /// @param _grid Grid object
  /// @param _num_elements number of particles (or points) being partitioned
  SpatialPartition(const Grid &_grid, const int _num_elements);

  /// @brief Destroy the Spatial Partition object
  SpatialPartition() = default;

  /// @brief Resets the memory of the Spatial Partition object
  void reset();

  /// @brief Calculates the cartesian hash of a set of points
  /// @param positions_gpu Set of points with the same size as _num_elements
  void calculate_hash(gpu_array<Vectorr> &positions_gpu);

  /// @brief Sorts hashes of the points
  void sort_hashes();

  /// @brief Bins the points into the grid
  void bin_particles();

  /// @brief start cell indices of the points
  gpu_array<int> cell_start_gpu;

  /// @brief end cell indices of the points
  gpu_array<int> cell_end_gpu;

  /// @brief sorted indices of the points
  gpu_array<int> sorted_index_gpu;

  ////@brief unsorted cartesian hashes of the coordinates
  gpu_array<unsigned int> hash_unsorted_gpu;

  /// @brief sorted cartesian hashes with respect to the sorted indices.
  gpu_array<unsigned int> hash_sorted_gpu;

  /// @brief bin ids of points within a cell, e.g  (x=1,y=2,z=3)
  gpu_array<Vectori> bins_gpu;

  /// @brief Grid object
  Grid grid = Grid();

  /// @brief number of points being partitioned
  int num_elements = 0;

#ifdef CUDA_ENABLED
  /// @brief GPU launch configuration for kernels
  GPULaunchConfig launch_config;
#endif
};
} // namespace pyroclastmpm