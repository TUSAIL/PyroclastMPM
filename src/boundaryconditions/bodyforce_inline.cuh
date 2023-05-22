

__device__ __host__ inline void apply_bodyforce(Vectorr *nodes_forces_external_gpu,
                                                const Vectorr *values_gpu,
                                                const bool *mask_gpu,
                                                const int node_mem_index)
{
    if (mask_gpu[node_mem_index])
    {
        nodes_forces_external_gpu[node_mem_index] += values_gpu[node_mem_index];
    }
}

#ifdef CUDA_ENABLED
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
    apply_bodyforce(nodes_forces_external_gpu, values_gpu, mask_gpu, node_mem_index);
}
#endif

__device__ __host__ inline void apply_bodymoments(
    Vectorr *nodes_moments_nt_gpu,
    Vectorr *nodes_moments_gpu,
    const Vectorr *values_gpu,
    const bool *mask_gpu,
    const bool isFixed,
    const int node_mem_index)
{

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

#ifdef CUDA_ENABLED
__global__ void KERNEL_APPLY_BODYMOMENT(Vectorr *nodes_moments_nt_gpu,
                                        Vectorr *nodes_moments_gpu,
                                        const Vectorr *values_gpu,
                                        const bool *mask_gpu,
                                        const bool isFixed,
                                        const int num_nodes_total)
{
    
    const int node_mem_index  = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
      return;
    }

    apply_bodymoments(nodes_moments_nt_gpu, nodes_moments_gpu, values_gpu, mask_gpu, isFixed, node_mem_index);
}
#endif