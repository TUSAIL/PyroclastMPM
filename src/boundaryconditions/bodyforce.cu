#include "pyroclastmpm/boundaryconditions/bodyforce.cuh"

namespace pyroclastmpm
{

#include "bodyforce_inline.cuh"

  BodyForce::BodyForce(const std::string _mode,
                       const cpu_array<Vectorr> _values,
                       const cpu_array<bool> _mask)
  {
    if (_mode == "forces")
    {
      mode_id = 0;
    }
    else if (_mode == "moments")
    {
      mode_id = 1;
    }
    else if (_mode == "fixed")
    {
      mode_id = 2;
    }

    set_default_device<Vectorr>(_values.size(), _values, values_gpu, Vectorr::Zero());
    set_default_device<bool>(_mask.size(), _mask, mask_gpu, false);

    // TODO check if window size is set
    // TODO add "checker for each class"
  }

  void BodyForce::apply_on_nodes_f_ext(NodesContainer &nodes_ref)
  {
    if (!isActive)
    {
      return;
    }

    if (mode_id == 0) // apply on external forces
    {
#ifdef CUDA_ENABLED
      KERNEL_APPLY_BODYFORCE<<<nodes_ref.launch_config.tpb,
                               nodes_ref.launch_config.bpg>>>(
          thrust::raw_pointer_cast(nodes_ref.forces_external_gpu.data()),
          thrust::raw_pointer_cast(values_gpu.data()),
          thrust::raw_pointer_cast(mask_gpu.data()), nodes_ref.num_nodes_total);
#else
      for (int nid = 0; nid < nodes_ref.num_nodes_total; nid++)
      {
        apply_bodyforce(nodes_ref.forces_external_gpu.data(),
                        values_gpu.data(),
                        mask_gpu.data(), nid);
      }

#endif
    }
  };

  void BodyForce::apply_on_nodes_moments(NodesContainer &nodes_ref,
                                         ParticlesContainer &particles_ref)
  {
    if (!isActive)
    {
      return;
    }

    bool isFixed = (mode_id == 2);

#ifdef CUDA_ENABLED
    KERNEL_APPLY_BODYMOMENT<<<nodes_ref.launch_config.tpb,
                              nodes_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
        thrust::raw_pointer_cast(nodes_ref.moments_gpu.data()),
        thrust::raw_pointer_cast(values_gpu.data()),
        thrust::raw_pointer_cast(mask_gpu.data()), isFixed, nodes_ref.num_nodes_total);
#else
    for (int nid = 0; nid < nodes_ref.num_nodes_total; nid++)
    {

      apply_bodymoments(nodes_ref.moments_nt_gpu.data(),
                       nodes_ref.moments_gpu.data(),
                       values_gpu.data(),
                       mask_gpu.data(), isFixed, nid);
    }
#endif
  };

} // namespace pyroclastmpm