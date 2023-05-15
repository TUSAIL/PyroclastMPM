
#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/shapefunctions/shapefunction_kernels.cuh"

namespace pyroclastmpm
{

  extern __constant__ SFType shape_function_gpu;
  extern __constant__ int num_surround_nodes_gpu;
  extern __constant__ int forward_window_gpu[64][3];

//  should be inlined
#include "./cubic.cuh"
#include "./linear.cuh"
#include "./quadratic.cuh"

  /**
   * @brief
   *
   * @param particles_dpsi_gpu Output gradient of the shape function
   * @param particles_psi_gpu Output shape function
   * @param particles_positions_gpu Particle positions
   * @param particles_bins_gpu Particle bins found by the partitioning algorithm
   * @param num_particles Total number of particles
   * @param num_surround_nodes Number of surrounding nodes per node (8 for linear
   * and 64 for quadratic)
   * @param num_cells Number of cells (relative to partitioning algorithm)
   * @param inv_cell_size Inverse cell size (relative to partitioning algorithm)
   * @param shape_function The selected shape function
   */
  __global__ void KERNEL_CALC_SHP(Vectorr *particles_dpsi_gpu,
                                  Real *particles_psi_gpu,
                                  const Vectorr *particles_positions_gpu,
                                  const Vectori *particles_bins_gpu,
                                  const Vectori *node_types_gpu,
                                  const Vectori num_cells,
                                  const Vectorr origin,
                                  const Real inv_cell_size,
                                  const int num_particles,
                                  const int num_nodes_total)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_particles)
    {
      return;
    }

    const Vectorr particle_coords = particles_positions_gpu[tid];
    const Vectori particle_bin = particles_bins_gpu[tid];

    for (int i = 0; i < num_surround_nodes_gpu; i++)
    {
    
#if DIM == 3
      const Vectori forward_mesh = Vectori({forward_window_gpu[i][0],
                                            forward_window_gpu[i][1],
                                            forward_window_gpu[i][2]});

      const Vectori selected_bin = particle_bin + forward_mesh;

      const unsigned int node_mem_index =
          selected_bin[0] + selected_bin[1] * num_cells[0] +
          selected_bin[2] * num_cells[0] * num_cells[1];
#elif DIM == 2
      const Vectori forward_mesh = Vectori({forward_window_gpu[i][0],
                                            forward_window_gpu[i][1]});

      const Vectori selected_bin = particle_bin + forward_mesh;

      const unsigned int node_mem_index =
          selected_bin[0] + selected_bin[1] * num_cells[0];

#else
      const Vectori forward_mesh = Vectori(forward_window_gpu[i][0]);

      const Vectori selected_bin = particle_bin + forward_mesh;

      const unsigned int node_mem_index = selected_bin[0];
#endif

      if (node_mem_index >= num_nodes_total)
      {
        continue;
      }
 
      Vectorr node_coords = Vectorr::Zero();

      const Vectorr relative_coordinates =
          (particle_coords - origin) * inv_cell_size - selected_bin.cast<Real>();


      Real psi_particle = 0.;

      Vectorr dpsi_particle = Vectorr::Zero();

      Vectorr N = Vectorr::Zero();
      Vectorr dN = Vectorr::Zero();

      Vectori node_type = node_types_gpu[node_mem_index];

      if (shape_function_gpu == LinearShapeFunction)
      {

        N[0] = linear(relative_coordinates[0]);
        dN[0] = derivative_linear(relative_coordinates[0], inv_cell_size);

#if DIM > 1
        N[1] = linear(relative_coordinates[1]);
        dN[1] = derivative_linear(relative_coordinates[1], inv_cell_size);
#endif

#if DIM > 2
        N[2] = linear(relative_coordinates[2]);
        dN[2] = derivative_linear(relative_coordinates[2], inv_cell_size);
#endif

      }
      else if (shape_function_gpu == QuadraticShapeFunction)
      {
      }
      else if (shape_function_gpu == CubicShapeFunction)
      {
      // printf("Linear SF! %f %f %f \n",relative_coordinates[0],relative_coordinates[1],relative_coordinates[2]);

        N[0] = cubic(relative_coordinates[0], node_type[0]);
        dN[0] = derivative_cubic(relative_coordinates[0], inv_cell_size,
                                 node_type[0]);
#if DIM > 1
        N[1] = cubic(relative_coordinates[1], node_type[1]);
        dN[1] = derivative_cubic(relative_coordinates[1], inv_cell_size,
                                 node_type[1]);
#endif
#if DIM > 2
        N[2] = cubic(relative_coordinates[2], node_type[2]);
        dN[2] = derivative_cubic(relative_coordinates[2], inv_cell_size,
                                 node_type[2]);
#endif
      }

#if DIM == 1
      psi_particle = N[0];
      dpsi_particle[0] = dN[0];
#elif DIM == 2
      psi_particle = N[0] * N[1];
      dpsi_particle[0] = dN[0] * N[1];
      dpsi_particle[1] = dN[1] * N[0];
#else
      psi_particle = N[0] * N[1] * N[2];
      dpsi_particle[0] = dN[0] * N[1] * N[2];
      dpsi_particle[1] = dN[1] * N[0] * N[2];
      dpsi_particle[2] = dN[2] * N[0] * N[1];
#endif

      particles_psi_gpu[tid * num_surround_nodes_gpu + i] = psi_particle;

      particles_dpsi_gpu[tid * num_surround_nodes_gpu + i] = dpsi_particle;
    }
  }

} // namespace pyroclastmpm