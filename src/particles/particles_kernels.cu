#include "pyroclastmpm/particles/particles_kernels.cuh"

namespace pyroclastmpm
{

  __global__ void KERNEL_CALCULATE_INITIAL_VOLUME(
      Real *particles_volumes_gpu,
      Real *particles_volumes_original_gpu,
      const Vectori *particles_bins_gpu,
      const int *particles_cells_start_gpu,
      const int *particles_cells_end_gpu,
      const Vectori num_cells,
      const Real cell_size,
      const int num_particles,
      const int num_nodes_total)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_particles)
    {
      return;
    }
    if (particles_volumes_gpu[tid] > 0.)
    {
      return;
    }
    const Vectori particle_bin = particles_bins_gpu[tid];

#if DIM == 3
    const unsigned int node_hash = particle_bin[0] +
                                   particle_bin[1] * num_cells[0] +
                                   particle_bin[2] * num_cells[0] * num_cells[1];
#elif DIM == 2
    const unsigned int node_hash = particle_bin[0] +
                                   particle_bin[1] * num_cells[0];

#else
    const unsigned int node_hash = particle_bin[0];
#endif

    if (node_hash >= num_nodes_total)
    {
      particles_volumes_gpu[tid] = 0.0;
      particles_volumes_original_gpu[tid] = 0.0;
      return;
    }

    const int cstart = particles_cells_start_gpu[node_hash];

    if (cstart < 0)
    {
      particles_volumes_gpu[tid] = 0.0;
      particles_volumes_original_gpu[tid] = 0.0;
      return;
    }

    const int cend = particles_cells_end_gpu[node_hash];

    if (cend < 0)
    {
      particles_volumes_gpu[tid] = 0.0;
      particles_volumes_original_gpu[tid] = 0.0;
      return;
    }

    const int particles_per_cell = cend - cstart;

#if DIM == 1
    Real volumes = cell_size;
#elif DIM == 2
    Real volumes = cell_size * cell_size;
#else
    Real volumes = cell_size * cell_size * cell_size;
#endif

    volumes /= particles_per_cell;

    particles_volumes_gpu[tid] = volumes;

    particles_volumes_original_gpu[tid] = volumes;
  }

} // namespace pyroclastmpm