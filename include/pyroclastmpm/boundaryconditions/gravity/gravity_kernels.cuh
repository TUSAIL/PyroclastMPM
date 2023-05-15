#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

/**
 * @brief This kernel applies gravitational force to nodes.
 *
 * @param nodes_forces_external_gpu Output external forces of the nodes
 * @param nodes_masses_gpu Masses of the nodes
 * @param gravity Gravity of the nodes
 * @param num_nodes Number of nodes
 */
__global__ void KERNEL_APPLY_GRAVITY(Vectorr* nodes_forces_external_gpu,
                                     const Real* nodes_masses_gpu,
                                     const Vectorr gravity,
                                     const int num_nodes_total);

}  // namespace pyroclastmpm