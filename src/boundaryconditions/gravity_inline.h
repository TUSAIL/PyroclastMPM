

__device__ __host__ inline void
apply_gravity(Vectorr *nodes_forces_external_gpu, const Real *nodes_masses_gpu,
              const Vectorr gravity, const int node_mem_index) {

  const Real node_mass = nodes_masses_gpu[node_mem_index];

  if (node_mass <= 0.000000001) {
    return;
  }

  nodes_forces_external_gpu[node_mem_index] += gravity * node_mass;
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_APPLY_GRAVITY(Vectorr *nodes_forces_external_gpu,
                                     const Real *nodes_masses_gpu,
                                     const Vectorr gravity,
                                     const int num_nodes_total)

{

  const int node_mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (node_mem_index >= num_nodes_total) {
    return;
  }

  apply_gravity(nodes_forces_external_gpu, nodes_masses_gpu, gravity,
                node_mem_index);
}

#endif