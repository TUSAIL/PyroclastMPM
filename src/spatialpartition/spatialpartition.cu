#include "pyroclastmpm/spatialpartition/spatialpartition.cuh"

namespace pyroclastmpm
{

  // extern int dimension_global_cpu;

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
  {

    grid_start = _node_start;

    grid_end = _node_end;

    cell_size = _node_spacing;

    inv_cell_size = 1. / cell_size;

    num_elements = _num_elements;

    num_cells_total = 1;

#if DIM == 3
    num_cells = Vectori({1, 1, 1});
#elif DIM == 2
    num_cells = Vectori({1, 1});
#else
    num_cells = Vectori(1);
#endif

    for (int axis = 0; axis < DIM; axis++)
    {
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

    launch_config.tpb = dim3(int((num_elements) / BLOCKSIZE) + 1, 1, 1);
    launch_config.bpg = dim3(BLOCKSIZE, 1, 1);

    gpuErrchk(cudaDeviceSynchronize());
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
  void SpatialPartition::reset()
  {
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
  void SpatialPartition::calculate_hash(
      gpu_array<Vectorr> &positions_gpu)
  {
    KERNEL_CALC_HASH<<<launch_config.tpb, launch_config.bpg>>>(
        thrust::raw_pointer_cast(bins_gpu.data()),
        thrust::raw_pointer_cast(hash_unsorted_gpu.data()),
        thrust::raw_pointer_cast(positions_gpu.data()), grid_start, grid_end,
        num_cells, inv_cell_size, num_elements);

    hash_sorted_gpu = hash_unsorted_gpu; // move this inside kernel?

    gpuErrchk(cudaDeviceSynchronize());
  }

  /**
   * @brief Sorts the particles by their hash value
   *
   */
  void SpatialPartition::sort_hashes()
  {
    thrust::stable_sort_by_key(hash_sorted_gpu.begin(), hash_sorted_gpu.end(),
                               sorted_index_gpu.begin());
    gpuErrchk(cudaDeviceSynchronize());
  }

  /**
   * @brief calculates the start and end of each cell in the grid containing the particles
   *
   */
  void SpatialPartition::bin_particles()
  {
    KERNEL_BIN_PARTICLES<<<launch_config.tpb, launch_config.bpg>>>(
        thrust::raw_pointer_cast(cell_start_gpu.data()),
        thrust::raw_pointer_cast(cell_end_gpu.data()),
        thrust::raw_pointer_cast(hash_sorted_gpu.data()), num_elements);

    gpuErrchk(cudaDeviceSynchronize());
  }

} // namespace pyroclastmpm