#pragma once

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "pyroclastmpm/common/helper.cuh"
#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/spatialpartition/spatialpartition_kernels.cuh"

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
  SpatialPartition(const Vectorr _node_start,
                   const Vectorr _node_end,
                   const Real _node_spacing,
                   const int _num_elements);

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
  void calculate_hash(gpu_array<Vectorr>& positions_gpu);

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

  GPULaunchConfig launch_config;
};
}  // namespace pyroclastmpm