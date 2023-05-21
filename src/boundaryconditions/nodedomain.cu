#include "pyroclastmpm/boundaryconditions/nodedomain.cuh"

namespace pyroclastmpm
{

    NodeDomain::NodeDomain(Vectori _axis0_mode, Vectori _axis1_mode)
    {
        axis0_mode = _axis0_mode;
        axis1_mode = _axis1_mode;
    }


  struct ApplyNodeDomain
  {
    Vectorr node_start;
    Vectorr node_end;
    Vectori num_nodes;
    Real inv_node_spacing;
    Vectori axis0_mode;
    Vectori axis1_mode;
    ApplyNodeDomain(
        const Vectorr _node_start,
        const Vectorr _node_end,
        const Vectori _num_nodes,
        const Real _inv_node_spacing,
        const Vectori _axis0_mode,
        const Vectori _axis1_mode) : node_start(_node_start),
                                    node_end(_node_end),
                                    num_nodes(_num_nodes),
                                    inv_node_spacing(_inv_node_spacing),
                                    axis0_mode(_axis0_mode),
                                    axis1_mode(_axis1_mode)
                                    {};

        template <typename Tuple>
        __host__ __device__ void operator()(Tuple tuple) const
        {
            Vectorr &moment_nt = thrust::get<0>(tuple);
            Vectorr &moment = thrust::get<1>(tuple);
            Real mass = thrust::get<2>(tuple);
            Vectori node_bin = thrust::get<3>(tuple);

            #pragma unroll
            for (int i = 0; i < DIM; i++)
            {

                if (node_bin[i] < 1)
                {

                    if (axis0_mode(i) == 0)
                    {
                        //stick
                        moment = Vectorr::Zero();
                        moment_nt = Vectorr::Zero();
                    }
                    else if (axis0_mode(i) == 1)
                    {
                        // slip
                        moment(i) = max(0., moment(i));
                        moment_nt(i) = max(0., moment_nt(i));
                    }
                }
                else if (node_bin[i] >= num_nodes(i) - 1)
                {
                    if (axis1_mode(i) == 0)
                    {   
                        //stick
                        moment = Vectorr::Zero();
                        moment_nt = Vectorr::Zero();
                    }
                    else if (axis1_mode(i) == 1)
                    {
                        // slip
                        moment(i) = max(0., moment(i));
                        moment_nt(i) = max(0.,moment_nt(i));
                    }
                }
            }


        }
    };
    void NodeDomain::apply_on_nodes_moments(NodesContainer &nodes_ref, ParticlesContainer &particles_ref)
    {
        // KERNEL_APPLY_NODEDOMAIN<<<nodes_ref.launch_config.tpb,
        //                           nodes_ref.launch_config.bpg>>>(
        //     thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
        //     thrust::raw_pointer_cast(nodes_ref.moments_gpu.data()),
        //     thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()),
        //     thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
        //     nodes_ref.node_start, nodes_ref.node_end, nodes_ref.num_nodes,
        //     nodes_ref.inv_node_spacing, axis0_mode, axis1_mode, nodes_ref.num_nodes_total);
        // gpuErrchk(cudaDeviceSynchronize());
    };


} // namespace pyroclastmpm