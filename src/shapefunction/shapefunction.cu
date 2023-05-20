#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/shapefunction/shapefunction.cuh"

namespace pyroclastmpm
{

#ifdef CUDA_ENABLED
      extern __constant__ SFType shape_function_gpu;
      extern __constant__ int num_surround_nodes_gpu;
      extern __constant__ int forward_window_gpu[64][3];
#else
      extern SFType shape_function_cpu;
      extern int num_surround_nodes_cpu;
      extern int forward_window_cpu[64][3];
#endif

//  should be inlined
#include "./cubic.cuh"
#include "./linear.cuh"
#include "./quadratic.cuh"

      __device__ __host__ inline void shape_function_kernel(
          Vectorr *particles_dpsi_gpu,
          Real *particles_psi_gpu,
          const Vectorr *particles_positions_gpu,
          const Vectori *particles_bins_gpu,
          const Vectori *node_types_gpu,
          const Vectori num_cells,
          const Vectorr origin,
          const Real inv_cell_size,
          const int num_nodes_total,
          const int tid)
      {

            const Vectorr particle_coords = particles_positions_gpu[tid];
            const Vectori particle_bin = particles_bins_gpu[tid];

#ifdef CUDA_ENABLED
            const int num_surround_nodes = num_surround_nodes_gpu;
            const SFType shape_function_type = shape_function_gpu;
#else
            const int num_surround_nodes = num_surround_nodes_cpu;
            const SFType shape_function_type = shape_function_cpu;
#endif
            for (int i = 0; i < num_surround_nodes; i++)
            {
// macros defined in types_common.cuh
#ifdef CUDA_ENABLED
                  const Vectori selected_bin = WINDOW_BIN(particle_bin, forward_window_gpu, i);
#else
                  const Vectori selected_bin = WINDOW_BIN(particle_bin, forward_window_cpu, i);
#endif

                  const unsigned int node_mem_index = NODE_MEM_INDEX(selected_bin, num_cells);

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

                  if (shape_function_type == LinearShapeFunction)
                  {
                        for (int axis = 0; axis < DIM; axis++)
                        {
                              N[axis] = linear(relative_coordinates[axis]);
                              dN[axis] = derivative_linear(relative_coordinates[axis], inv_cell_size);
                        }
                  }
                  else if (shape_function_type == QuadraticShapeFunction)
                  {
                  }
                  else if (shape_function_type == CubicShapeFunction)
                  {

                        printf("Node types not implemented :( fix this first \n");
                        for (int axis = 0; axis < DIM; axis++)
                        {
                              N[axis] = cubic(relative_coordinates[axis], node_type[axis]);
                              dN[axis] = derivative_cubic(relative_coordinates[axis], inv_cell_size, node_type[axis]);
                        }
                        printf(" N %f %f  \n", N[0], N[1]);
                  }
                  else
                  {
                        printf("Shape function not implemented\n");
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
                  particles_psi_gpu[tid * num_surround_nodes + i] = psi_particle;
                  particles_dpsi_gpu[tid * num_surround_nodes + i] = dpsi_particle;
            }
      }

#ifdef CUDA_ENABLED
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
            shape_function_kernel(particles_dpsi_gpu,
                                  particles_psi_gpu,
                                  particles_positions_gpu,
                                  particles_bins_gpu,
                                  node_types_gpu,
                                  num_cells,
                                  origin,
                                  inv_cell_size,
                                  num_nodes_total,
                                  tid);
      }
#endif

      void calculate_shape_function(NodesContainer &nodes_ref, ParticlesContainer &particles_ref)
      {

#ifdef CUDA_ENABLED
            KERNEL_CALC_SHP<<<particles_ref.launch_config.tpb, particles_ref.launch_config.bpg>>>(
                thrust::raw_pointer_cast(particles_ref.dpsi_gpu.data()),
                thrust::raw_pointer_cast(particles_ref.psi_gpu.data()),
                thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
                thrust::raw_pointer_cast(particles_ref.spatial.bins_gpu.data()),
                thrust::raw_pointer_cast(nodes_ref.node_types_gpu.data()), nodes_ref.num_nodes,
                nodes_ref.node_start, nodes_ref.inv_node_spacing, particles_ref.num_particles,
                nodes_ref.num_nodes_total);
            gpuErrchk(cudaDeviceSynchronize());

#else
            for (size_t pi = 0; pi < particles_ref.num_particles; pi++)
            {
                  shape_function_kernel(particles_ref.dpsi_gpu.data(),
                                        particles_ref.psi_gpu.data(),
                                        particles_ref.positions_gpu.data(),
                                        particles_ref.spatial.bins_gpu.data(),
                                        nodes_ref.node_types_gpu.data(),
                                        nodes_ref.num_nodes,
                                        nodes_ref.node_start,
                                        nodes_ref.inv_node_spacing,
                                        nodes_ref.num_nodes_total,
                                        pi);
            }
#endif
      }

} // namespace pyroclastmpm