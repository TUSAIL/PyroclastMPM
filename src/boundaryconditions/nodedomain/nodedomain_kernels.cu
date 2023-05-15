#include "pyroclastmpm/boundaryconditions/noslipwall/noslipwall_kernels.cuh"

namespace pyroclastmpm
{

    // extern __constant__ int window_size_gpu;

    __global__ void KERNEL_APPLY_NODEDOMAIN(Vectorr *nodes_moments_nt_gpu,
                                            Vectorr *nodes_moments_gpu,
                                            Real *nodes_masses_gpu,
                                            const Vectori *nodes_bins_gpu,
                                            const Vectorr node_start,
                                            const Vectorr node_end,
                                            const Vectori num_nodes,
                                            const Real inv_node_spacing,
                                            const Vectori axis0_mode,
                                            const Vectori axis1_mode,
                                            const int num_nodes_total)
    {
        const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (node_mem_index >= num_nodes_total)
        {
            return;
        }

        const Vectori node_bin = nodes_bins_gpu[node_mem_index];

        for (int i = 0; i < DIM; i++)
        {
            // axis0axis0_mode
            if (node_bin[i] < 1)
            {

                if (axis0_mode(i) == 0)
                {
                    nodes_moments_gpu[node_mem_index] = Vectorr::Zero();
                    nodes_moments_nt_gpu[node_mem_index] = Vectorr::Zero();
                }
                else if (axis0_mode(i) == 1)
                {
                    // slip
                    nodes_moments_gpu[node_mem_index][i] =
                        max(0., nodes_moments_gpu[node_mem_index][i]);
                    nodes_moments_nt_gpu[node_mem_index][i] =
                        max(0., nodes_moments_nt_gpu[node_mem_index][i]);
                }
            }
            else if (node_bin[i] >= num_nodes(i) - 1)
            {
                if (axis1_mode(i) == 0)
                {
                    nodes_moments_gpu[node_mem_index] = Vectorr::Zero();
                    nodes_moments_nt_gpu[node_mem_index] = Vectorr::Zero();
                }
                else if (axis1_mode(i) == 1)
                {
                    // slip
                    nodes_moments_gpu[node_mem_index][i] =
                        max(0., nodes_moments_gpu[node_mem_index][i]);
                    nodes_moments_nt_gpu[node_mem_index][i] =
                        max(0., nodes_moments_nt_gpu[node_mem_index][i]);
                }
            }
        }

        // __global__ void KERNEL_APPLY_NOSLIPWALL(Vectorr* nodes_moments_nt_gpu,
        //                                       Vectorr* nodes_moments_gpu,
        //                                       Real* nodes_masses_gpu,
        //                                       const Vectori * nodes_bins_gpu,
        //                                       const Vectorr node_start,
        //                                       const Vectorr node_end,
        //                                       const Vectori num_nodes,
        //                                       const Real inv_node_spacing,
        //                                       const int axis_key,
        //                                       const int plane_key,
        //                                       const int num_nodes_total
        //                                       ) {
        //     const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

        //     if (node_mem_index >= num_nodes_total)
        //     {
        //       return;
        //     }

        //   const Vectori node_bin = nodes_bins_gpu[node_mem_index];

        //   if (plane_key == 0) {
        //     if (node_bin[axis_key] >= window_size_gpu) {
        //       return;
        //     }

        //   } else if (plane_key == 1) {
        //     if (node_bin[axis_key] < num_nodes[axis_key] - window_size_gpu) {
        //       return;
        //     }

        //     nodes_moments_gpu[node_mem_index] = Vectorr::Zero();
        //     nodes_moments_nt_gpu[node_mem_index] = Vectorr::Zero();

        //   } else {
        //     return;
        //   }
    }
} // namespace pyroclastmpm