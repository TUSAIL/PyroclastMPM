

__device__ __host__ inline void integrate_nodes(Vectorr *nodes_moments_nt_gpu,
                                                Vectorr *nodes_forces_total_gpu,
                                                const Vectorr *nodes_forces_external_gpu,
                                                const Vectorr *nodes_forces_internal_gpu,
                                                const Vectorr *nodes_moments_gpu,
                                                const Real *nodes_masses_gpu,
                                                const int node_mem_index)
{

    if (nodes_masses_gpu[node_mem_index] <= 0.000000001)
    {
        return;
    }
    const Vectorr ftotal = nodes_forces_internal_gpu[node_mem_index] +
                           nodes_forces_external_gpu[node_mem_index];

    nodes_forces_total_gpu[node_mem_index] = ftotal;

#ifdef CUDA_ENABLED
    nodes_moments_nt_gpu[node_mem_index] =
        nodes_moments_gpu[node_mem_index] + ftotal * dt_gpu;
#else
    nodes_moments_nt_gpu[node_mem_index] =
        nodes_moments_gpu[node_mem_index] + ftotal * dt_cpu;
#endif
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_INTEGRATE(Vectorr *nodes_moments_nt_gpu,
                                 Vectorr *nodes_forces_total_gpu,
                                 const Vectorr *nodes_forces_external_gpu,
                                 const Vectorr *nodes_forces_internal_gpu,
                                 const Vectorr *nodes_moments_gpu,
                                 const Real *nodes_masses_gpu,
                                 const int num_nodes_total)
{

    const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_mem_index >= num_nodes_total)
    {
        return;
    }

    integrate_nodes(nodes_moments_nt_gpu,
                    nodes_forces_total_gpu,
                    nodes_forces_external_gpu,
                    nodes_forces_internal_gpu,
                    nodes_moments_gpu,
                    nodes_masses_gpu,
                    node_mem_index);
}

#endif