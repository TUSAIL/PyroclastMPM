#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm
{

    __global__ void KERNEL_APPLY_BODYFORCE(Vectorr *nodes_forces_external_gpu,
                                           const Vectorr *values_gpu,
                                           const bool *mask_gpu,
                                           const int num_nodes_total);

    __global__ void KERNEL_APPLY_BODYMOMENT(Vectorr *nodes_moments_nt_gpu,
                                            Vectorr *nodes_moments_gpu,
                                            const Vectorr *values_gpu,
                                            const bool *mask_gpu,
                                            const bool isFixed,
                                            const int num_nodes_total);
} // namespace pyroclastmpm