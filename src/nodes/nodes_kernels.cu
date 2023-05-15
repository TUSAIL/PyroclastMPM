#include "pyroclastmpm/nodes/nodes_kernels.cuh"

namespace pyroclastmpm
{

  extern __constant__ Real dt_gpu;

  extern __constant__ int num_surround_nodes_gpu;

  extern __constant__ SFType shape_function_gpu;

  /**
   * @brief This kernel sets the total force and integrates the force to find
   * the moment.
   *
   * @param nodes_moments_nt_gpu output nodal moments at next incremental step
   * (USL)
   * @param nodes_forces_total_gpu  output total nodal forces
   * @param nodes_forces_external_gpu nodal external forces (gravity)
   * @param nodes_forces_internal_gpu nodal internal forces ( from stress )
   * @param nodes_moments_gpu nodal moment at currentl incremental step
   * @param nodes_masses_gpu nodal mass
   * @param dt time step
   * @param num_nodes number of nodes in grid dimensions
   */
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

    if (nodes_masses_gpu[node_mem_index] <= 0.000000001)
    {
      return;
    }

    const Vectorr ftotal = nodes_forces_internal_gpu[node_mem_index] +
                           nodes_forces_external_gpu[node_mem_index];

    nodes_forces_total_gpu[node_mem_index] = ftotal;

    nodes_moments_nt_gpu[node_mem_index] =
        nodes_moments_gpu[node_mem_index] + ftotal * dt_gpu;
  }

  __global__ void KERNEL_GIVE_NODE_COORDS(Vectorr *nodes_coords_gpu,
                                          const Real inv_node_spacing,
                                          const Vectori num_nodes)
  {
#if DIM == 1
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const Vectori node_bin = Vectori(tid_x);
#elif DIM == 2
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
    const Vectori node_bin = Vectori({tid_x, tid_y});
#else
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tid_z = blockDim.z * blockIdx.z + threadIdx.z;
    const Vectori node_bin = Vectori({tid_x, tid_y, tid_z});
#endif

#pragma unroll
    for (int axis = 0; axis < DIM; axis++)
    {
      if (node_bin(axis) >= num_nodes(axis))
      {
        return;
      }
    }

#if DIM == 1
    const unsigned int node_mem_index = node_bin[0];
#elif DIM == 2
    const unsigned int node_mem_index = node_bin[0] + node_bin[1] * num_nodes[0];
#else
    const unsigned int node_mem_index = node_bin[0] + node_bin[1] * num_nodes[0] +
                                        node_bin[2] * num_nodes[0] * num_nodes[1];
#endif

#pragma unroll
    for (int axis = 0; axis < DIM; axis++)
    {
      nodes_coords_gpu[node_mem_index][axis] =
          ((Real)node_bin[axis]) / inv_node_spacing;
    }
  }

  __global__ void KERNEL_GIVE_NODE_IDS(Vectori *node_ids_gpu,
                                       const Vectori num_nodes)
  {

#if DIM == 1
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const Vectori node_bin = Vectori(tid_x);
#elif DIM == 2
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
    const Vectori node_bin = Vectori({tid_x, tid_y});
#else
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tid_z = blockDim.z * blockIdx.z + threadIdx.z;
    const Vectori node_bin = Vectori({tid_x, tid_y, tid_z});
#endif

#pragma unroll
    for (int axis = 0; axis < DIM; axis++)
    {
      if (node_bin(axis) >= num_nodes(axis))
      {
        return;
      }
    }

#if DIM == 1
    const unsigned int node_mem_index = node_bin[0];
#elif DIM == 2
    const unsigned int node_mem_index = node_bin[0] + node_bin[1] * num_nodes[0];
#else
    const unsigned int node_mem_index = node_bin[0] + node_bin[1] * num_nodes[0] +
                                        node_bin[2] * num_nodes[0] * num_nodes[1];
#endif

    node_ids_gpu[node_mem_index] = node_bin;
  }

  __global__ void KERNEL_SET_NODE_TYPES(Vectori *node_types_gpu,
                                        const Vectori num_nodes)
  {
    //TODO CHECK IF THIS KERNEL IS CORRECT

#if DIM == 1
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const Vectori node_bin = Vectori(tid_x);
#elif DIM == 2
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
    const Vectori node_bin = Vectori({tid_x, tid_y});
#else
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tid_z = blockDim.z * blockIdx.z + threadIdx.z;
    const Vectori node_bin = Vectori({tid_x, tid_y, tid_z});
#endif

#pragma unroll
    for (int axis = 0; axis < DIM; axis++)
    {
      if (node_bin(axis) >= num_nodes(axis))
      {
        return;
      }
    }

#if DIM == 1
    const unsigned int node_mem_index = node_bin[0];
#elif DIM == 2
    const unsigned int node_mem_index = node_bin[0] + node_bin[1] * num_nodes[0];
#else
    const unsigned int node_mem_index = node_bin[0] + node_bin[1] * num_nodes[0] +
                                        node_bin[2] * num_nodes[0] * num_nodes[1];
#endif

    Vectori node_type = Vectori::Zero();

// see page 45 B-splines 25year review
#pragma unroll
    for (int axis = 0; axis < DIM; axis++)
    {
      if (shape_function_gpu == CubicShapeFunction)
      {
        if ((node_bin[axis] == 0) | (node_bin[axis] == num_nodes[axis] - 1))
        {
          // Cell at boundary
          node_type[axis] = 1;
        }
        else if (node_bin[axis] == 1)
        {
          // Cell right of boundary
          node_type[axis] = 2;
        }
        else if (node_bin[axis] == num_nodes[axis] - 2)
        {
          // Cell left of boundary
          node_type[axis] = 4;
        }
        else
        {
          node_type[axis] = 3;
        }
      }
    }

    node_types_gpu[node_mem_index] = node_type;
  }

  // write an algorithm to do Newmark implicit integration

} // namespace pyroclastmpm