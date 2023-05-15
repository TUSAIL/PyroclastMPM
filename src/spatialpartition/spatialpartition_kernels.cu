
#include "pyroclastmpm/spatialpartition/spatialpartition_kernels.cuh"

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
   * @param num_cells Number of grid cells (Nx,Ny,Nz)
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
                                   const int num_elements)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_elements)
    {
      return;
    } // block access threads

    const Vectorr relative_position =
        (positions_gpu[tid] - grid_start) * inv_cell_size;

#if DIM == 3
    const Vectori bin = Vectori(
        (int)floor(relative_position(0)),
        (int)floor(relative_position(1)),
        (int)floor(relative_position(2)));
    uint hash = bin(0) + bin(1) * num_cells(0) + bin(2) * num_cells(0) * num_cells(1);
#elif DIM == 2
    const Vectori bin = Vectori(
        (int)floor(relative_position(0)),
        (int)floor(relative_position(1)));
    uint hash = bin(0) + bin(1) * num_cells(0);
#else
    const Vectori bin = Vectori((int)floor(relative_position(0)));
    uint hash = bin(0);
#endif

    hashes_unsorted_gpu[tid] = hash;
    bins_gpu[tid] = bin;
  }

  /**
   * @brief Bins the (sorted) primitives in to cells
   *
   * @param cell_start Output cell start or bin containing start indices of
   * primitives
   * @param cell_end  Output cell end or bin containing end indices of primitives
   * @param hashes_sorted Sorted hashes
   * @param num_elements Number of primitives
   */
  __global__ void KERNEL_BIN_PARTICLES(int *cell_start,
                                       int *cell_end,
                                       const unsigned int *hashes_sorted,
                                       const int num_elements)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

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
} // namespace pyroclastmpm
