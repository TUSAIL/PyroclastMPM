#include "pyroclastmpm/spatialpartition/spatialpartition.cuh"

namespace pyroclastmpm
{

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
      : grid_start(_node_start),
        grid_end(_node_end),
        cell_size(_node_spacing),
        num_elements(_num_elements)

  {
    inv_cell_size = 1. / cell_size;
    num_cells_total = 1;
    num_cells = Vectori::Ones();

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
  void SpatialPartition::reset()
  {
    thrust::sequence(sorted_index_gpu.begin(), sorted_index_gpu.end(), 0, 1);
    thrust::fill(cell_start_gpu.begin(), cell_start_gpu.end(), -1);
    thrust::fill(cell_end_gpu.begin(), cell_end_gpu.end(), -1);
    thrust::fill(hash_sorted_gpu.begin(), hash_sorted_gpu.end(), 0);
    thrust::fill(hash_unsorted_gpu.begin(), hash_unsorted_gpu.end(), 0);
    thrust::fill(bins_gpu.begin(), bins_gpu.end(), Vectori::Zero());
  }

  struct CalculateHash
  {
    Vectorr grid_start;
    Vectorr grid_end;
    Vectori num_cells;
    Real inv_cell_size;
    CalculateHash(
        const Vectorr _grid_start,
        const Vectorr _grid_end,
        const Vectori _num_cells,
        const Real _inv_cell_size) : grid_start(_grid_start),
                                     grid_end(_grid_end),
                                     num_cells(_num_cells),
                                     inv_cell_size(_inv_cell_size){};

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple tuple) const
    {
      Vectori &bin = thrust::get<0>(tuple);
      unsigned int &hash_unsorted = thrust::get<1>(tuple);
      Vectorr position = thrust::get<2>(tuple);

      const Vectorr relative_position = (position - grid_start) * inv_cell_size;
      bin = relative_position.cast<int>();
      hash_unsorted = NODE_MEM_INDEX(bin, num_cells); // MACRO defined in type_commons.cuh
    }
  };

  /**
   * @brief Sorts the particles by their hash value
   *
   * @param positions_gpu
   */
  void SpatialPartition::calculate_hash(
      gpu_array<Vectorr> &positions_gpu)
  {
    execution_policy exec;

    PARALLEL_FOR_EACH_ZIP(exec,
                          num_elements,
                          CalculateHash(grid_start,
                                        grid_end,
                                        num_cells,
                                        inv_cell_size),
                          bins_gpu.begin(),
                          hash_unsorted_gpu.begin(),
                          positions_gpu.begin());

    hash_sorted_gpu = hash_unsorted_gpu; // move this inside kernel?
  }

  /**
   * @brief Sorts the particles by their hash value
   *
   */
  void SpatialPartition::sort_hashes()
  {
    thrust::stable_sort_by_key(hash_sorted_gpu.begin(), hash_sorted_gpu.end(),
                               sorted_index_gpu.begin());
  }

  __host__ __device__ inline void bin_particles_kernel(
      int *cell_start,
      int *cell_end,
      const unsigned int *hashes_sorted,
      const int num_elements,
      const int tid)
  {

    if (tid >= num_elements)
    {
      return;
    } // block access threads

    unsigned int hash, nexthash;
    hash = hashes_sorted[tid];

    if (tid < num_elements - 1)
    {
      nexthash = hashes_sorted[tid + 1];

      if (tid == 0)
      {
        cell_start[hash] = tid;
      }

      if (hash != nexthash)
      {
        cell_end[hash] = tid + 1;

        cell_start[nexthash] = tid + 1;
      }
    }

    if (tid == num_elements - 1)
    {
      cell_end[hash] = tid + 1;
    }
  }

#ifdef CUDA_ENABLED
  __global__ void KERNEL_BIN_PARTICLES(int *cell_start,
                                       int *cell_end,
                                       const unsigned int *hashes_sorted,
                                       const int num_elements)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    bin_particles_kernel(cell_start, cell_end, hashes_sorted, num_elements, tid);
  }
#endif

  /**
   * @brief calculates the start and end of each cell in the grid containing the particles
   *
   */
  void SpatialPartition::bin_particles()
  {
#ifdef CUDA_ENABLED
    KERNEL_BIN_PARTICLES<<<launch_config.tpb, launch_config.bpg>>>(
        thrust::raw_pointer_cast(cell_start_gpu.data()),
        thrust::raw_pointer_cast(cell_end_gpu.data()),
        thrust::raw_pointer_cast(hash_sorted_gpu.data()), num_elements);
    gpuErrchk(cudaDeviceSynchronize());
#else
    for (size_t ti = 0; ti < num_elements; ti++)
    {
      bin_particles_kernel(
          cell_start_gpu.data(),
          cell_end_gpu.data(),
          hash_sorted_gpu.data(),
          num_elements,
          ti);
    }
#endif
  }

} // namespace pyroclastmpm