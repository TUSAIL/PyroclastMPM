

__device__ __host__ inline void
calculate_hashes(Vectori *bins_gpu, unsigned int *hashes_unsorted_gpu,
                 const Vectorr *positions_gpu, const Vectorr grid_start,
                 const Vectorr grid_end, const Vectori num_cells,
                 const Real inv_cell_size, const int tid)

{
  const Vectorr relative_position =
      (positions_gpu[tid] - grid_start) * inv_cell_size;
  bins_gpu[tid] = relative_position.cast<int>();
  hashes_unsorted_gpu[tid] = NODE_MEM_INDEX(
      bins_gpu[tid], num_cells); // MACRO defined in type_commons.cuh
}

#ifdef CUDA_ENABLED
__global__ void
KERNEL_CALC_HASH(Vectori *bins_gpu, unsigned int *hashes_unsorted_gpu,
                 const Vectorr *positions_gpu, const Vectorr grid_start,
                 const Vectorr grid_end, const Vectori num_cells,
                 const Real inv_cell_size, const int num_elements) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_elements) {
    return;
  } // block access threads

  calculate_hashes(bins_gpu, hashes_unsorted_gpu, positions_gpu, grid_start,
                   grid_end, num_cells, inv_cell_size, tid);
}

#endif

__host__ __device__ inline void
bin_particles_kernel(int *cell_start, int *cell_end,
                     const unsigned int *hashes_sorted, const int num_elements,
                     const int tid) {

  if (tid >= num_elements) {
    return;
  } // block access threads

  unsigned int hash, nexthash;
  hash = hashes_sorted[tid];

  if (tid < num_elements - 1) {
    nexthash = hashes_sorted[tid + 1];

    if (tid == 0) {
      cell_start[hash] = tid;
    }

    if (hash != nexthash) {
      cell_end[hash] = tid + 1;

      cell_start[nexthash] = tid + 1;
    }
  }

  if (tid == num_elements - 1) {
    cell_end[hash] = tid + 1;
  }
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_BIN_PARTICLES(int *cell_start, int *cell_end,
                                     const unsigned int *hashes_sorted,
                                     const int num_elements) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  bin_particles_kernel(cell_start, cell_end, hashes_sorted, num_elements, tid);
}
#endif