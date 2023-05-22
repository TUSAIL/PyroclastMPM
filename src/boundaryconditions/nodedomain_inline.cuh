
__device__ __host__ inline void apply_nodedomain(Vectorr *nodes_moments_nt_gpu,
                                                 Vectorr *nodes_moments_gpu,
                                                 Real *nodes_masses_gpu,
                                                 const Vectori *nodes_bins_gpu,
                                                 const Vectorr node_start,
                                                 const Vectorr node_end,
                                                 const Vectori num_nodes,
                                                 const Real inv_node_spacing,
                                                 const Vectori axis0_mode,
                                                 const Vectori axis1_mode,
                                                 const int node_mem_index)
{

#ifndef CUDA_ENABLED
            // to call std::max on CPU and avoid error occuring without max:
            // `there are no arguments to 'max' that depend on a template parameter...`
            using namespace std;
#endif

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
                    max(0., nodes_moments_gpu[node_mem_index].cast<double>()[i]); // cast to double to avoid error with std::max
                nodes_moments_nt_gpu[node_mem_index][i] =
                    max(0., nodes_moments_nt_gpu[node_mem_index].cast<double>()[i]);
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
                    max(0., nodes_moments_gpu[node_mem_index].cast<double>()[i]);
                nodes_moments_nt_gpu[node_mem_index][i] =
                    max(0., nodes_moments_nt_gpu[node_mem_index].cast<double>()[i]);
            }
        }
    }
}


#ifdef CUDA_ENABLED
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

    apply_nodedomain(nodes_moments_nt_gpu,
                     nodes_moments_gpu,
                     nodes_masses_gpu,
                     nodes_bins_gpu,
                     node_start,
                     node_end,
                     num_nodes,
                     inv_node_spacing,
                     axis0_mode,
                     axis1_mode,
                     node_mem_index);
}

#endif 