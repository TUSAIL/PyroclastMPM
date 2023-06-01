#include "pyroclastmpm/boundaryconditions/nodedomain.h"

namespace pyroclastmpm {

#include "nodedomain_inline.h"

NodeDomain::NodeDomain(Vectori _axis0_mode, Vectori _axis1_mode) {
  axis0_mode = _axis0_mode;
  axis1_mode = _axis1_mode;
}

void NodeDomain::apply_on_nodes_moments(NodesContainer &nodes_ref,
                                        ParticlesContainer &particles_ref) {

#ifdef CUDA_ENABLED
  KERNEL_APPLY_NODEDOMAIN<<<nodes_ref.launch_config.tpb,
                            nodes_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.moments_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
      nodes_ref.node_start, nodes_ref.node_end, nodes_ref.num_nodes,
      nodes_ref.inv_node_spacing, axis0_mode, axis1_mode,
      nodes_ref.num_nodes_total);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int nid = 0; nid < nodes_ref.num_nodes_total; nid++) {

    apply_nodedomain(nodes_ref.moments_nt_gpu.data(),
                     nodes_ref.moments_gpu.data(), nodes_ref.masses_gpu.data(),
                     nodes_ref.node_ids_gpu.data(), nodes_ref.node_start,
                     nodes_ref.node_end, nodes_ref.num_nodes,
                     nodes_ref.inv_node_spacing, axis0_mode, axis1_mode, nid);
  }
#endif
};

} // namespace pyroclastmpm