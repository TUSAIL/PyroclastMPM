#include "pyroclastmpm/boundaryconditions/gravity/gravity_kernels.cuh"

namespace pyroclastmpm
{

  /**
   * @brief This kernel applies gravitational force to nodes.
   *
   * @param nodes_forces_external_gpu Output external forces of the nodes
   * @param nodes_masses_gpu Masses of the nodes
   * @param gravity Gravity of the nodes
   * @param num_nodes Number of nodes
   */
  __global__ void KERNEL_APPLY_GRAVITY(Vectorr *nodes_forces_external_gpu,
                                       const Real *nodes_masses_gpu,
                                       const Vectorr gravity,
                                       const int num_nodes_total)

  {

    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    const Real node_mass = nodes_masses_gpu[node_mem_index];

    if (node_mass <= 0.000000001) {
      return;
    }
    
    nodes_forces_external_gpu[node_mem_index] += gravity * node_mass;
    
  }

} // namespace pyroclastmpm
