#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm
{

    __global__ void KERNEL_INTEGRATE(Vectorr *nodes_moments_nt_gpu,
                                     Vectorr *nodes_forces_total_gpu,
                                     const Vectorr *nodes_forces_external_gpu,
                                     const Vectorr *nodes_forces_internal_gpu,
                                     const Vectorr *nodes_moments_gpu,
                                     const Real *nodes_masses_gpu,
                                     const int num_nodes_total);

    __global__ void KERNEL_GIVE_NODE_COORDS(Vectorr *nodes_coords_gpu,
                                            const Real inv_node_spacing,
                                            const Vectori num_nodes);

    __global__ void KERNEL_GIVE_NODE_IDS(Vectori *node_ids_gpu,
                                            const Vectori num_nodes);

    __global__ void KERNEL_SET_NODE_TYPES(Vectori *node_types_gpu,
                                          const Vectori num_nodes);

} // namespace pyroclastmpm