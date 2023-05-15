#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

__global__ void KERNEL_APPLY_PERIODICWALL_NODES(Vectorr* nodes_moments_nt_gpu,
                                                Vectorr* nodes_moments_gpu,
                                                Real* nodes_masses_gpu,
                                                const Vectori* nodes_bins_gpu,
                                                const Vectorr node_start,
                                                const Vectorr node_end,
                                                const Vectori num_nodes,
                                                const Real inv_node_spacing,
                                                const int axis_key,
                                                const int num_nodes_total
                                                );

__global__ void KERNEL_APPLY_PERIODICWALL_PARTICLES(
    Vectorr* particles_positions_gpu,
    const Vectorr start_domain,
    const Vectorr end_domain,
    const Real node_spacing,
    const int axis_key,
    const int num_particles);
    
}