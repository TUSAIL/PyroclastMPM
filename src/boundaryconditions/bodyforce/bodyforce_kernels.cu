#include "pyroclastmpm/boundaryconditions/bodyforce/bodyforce_kernels.cuh"

namespace pyroclastmpm
{

  extern __constant__ int dimension_global_gpu;

  /**
   * @brief This kernel applies gravitational force to nodes.
   *
   * @param nodes_forces_external_gpu Output external forces of the nodes
   * @param nodes_masses_gpu Masses of the nodes
   * @param gravity Gravity of the nodes
   * @param num_nodes Number of nodes
   */
  __global__ void KERNEL_APPLY_BODYFORCE(Vectorr *nodes_forces_external_gpu,
                                         const Vectorr *values_gpu,
                                         const bool *mask_gpu,
                                         const int num_nodes_total)

  {

    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    if (mask_gpu[node_mem_index])
    {
      nodes_forces_external_gpu[node_mem_index] += values_gpu[node_mem_index];
    }
  }

  __global__ void KERNEL_APPLY_BODYMOMENT(Vectorr *nodes_moments_nt_gpu,
                                          Vectorr *nodes_moments_gpu,
                                          const Vectorr *values_gpu,
                                          const bool *mask_gpu,
                                          const bool isFixed,
                                          const int num_nodes_total)
  {
    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    if (mask_gpu[node_mem_index])
    {
      if (isFixed)
      {
        nodes_moments_gpu[node_mem_index] = values_gpu[node_mem_index];
        nodes_moments_nt_gpu[node_mem_index] = values_gpu[node_mem_index];
      }
      else
      {
        nodes_moments_gpu[node_mem_index] += values_gpu[node_mem_index];

        // TODO check if nodes_moments_nt needs to be incremented?
      }
    }
  }
} // namespace pyroclastmpm