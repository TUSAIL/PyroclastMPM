#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

__global__ void KERNEL_APPLY_NOSLIPWALL(Vectorr* nodes_moments_nt_gpu,
                                      Vectorr* nodes_moments_gpu,
                                      Real* nodes_masses_gpu,
                                      const Vectori * nodes_bins_gpu,
                                      const Vectorr node_start,
                                      const Vectorr node_end,
                                      const Vectori num_nodes,
                                      const Real inv_node_spacing,
                                      const int axis_key,
                                      const int plane_key,
                                      const int num_nodes_total
                                      );

}