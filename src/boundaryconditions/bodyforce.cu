#include "pyroclastmpm/boundaryconditions/bodyforce.cuh"

namespace pyroclastmpm
{

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

  struct ApplyBodyForce
  {
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple tuple) const
    {

      Vectorr &f_ext = thrust::get<0>(tuple);
      const Vectorr value = thrust::get<1>(tuple);
      const bool mask = thrust::get<2>(tuple);

      if (mask)
      {
        f_ext += value;
      }
    }
  };

  void BodyForce::apply_on_nodes_f_ext(NodesContainer &nodes_ref)
  {
    if (!isActive)
    {
      return;
    }

    if (mode_id == 0) // apply on external forces
    {
      execution_policy exec;
      PARALLEL_FOR_EACH_ZIP(exec,
                            nodes_ref.num_nodes_total,
                            ApplyBodyForce(),
                            nodes_ref.forces_external_gpu.data(),
                            values_gpu.data(),
                            mask_gpu.data());
    }
  };

  struct ApplyBodyMoments
  {

    bool isFixed;
    ApplyBodyMoments(bool _isFixed) : isFixed(_isFixed){};

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple tuple) const
    {
      Vectorr &moment_nt = thrust::get<0>(tuple);
      Vectorr &moment = thrust::get<1>(tuple);
      const Vectorr value = thrust::get<2>(tuple);
      const bool mask = thrust::get<3>(tuple);

      if (mask)
      {
        if (isFixed)
        {
          moment = value;
          moment_nt = value;
        }
        else
        {
          moment += value;

          // TODO check if nodes_moments_nt needs to be incremented?
        }
      }
    }
  };

  void BodyForce::apply_on_nodes_moments(NodesContainer &nodes_ref,
                                         ParticlesContainer &particles_ref)
  {
    if (!isActive)
    {
      return;
    }

    if (mode_id == 1) // apply on moment
    {

      execution_policy exec;
      PARALLEL_FOR_EACH_ZIP(exec,
                            nodes_ref.num_nodes_total,
                            ApplyBodyMoments(false),
                            nodes_ref.moments_nt_gpu.data(),
                            nodes_ref.moments_gpu.data(),
                            values_gpu.data(),
                            mask_gpu.data());
    }
    else if (mode_id == 2) // fixed moment
    {

      execution_policy exec;
      PARALLEL_FOR_EACH_ZIP(exec,
                            nodes_ref.num_nodes_total,
                            ApplyBodyMoments(true),
                            nodes_ref.moments_nt_gpu.data(),
                            nodes_ref.moments_gpu.data(),
                            values_gpu.data(),
                            mask_gpu.data());
    }
  };

} // namespace pyroclastmpm