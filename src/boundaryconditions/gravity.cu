#include "pyroclastmpm/boundaryconditions/gravity.cuh"

namespace pyroclastmpm {


extern int global_step_cpu;

/**
 * @brief Gravity boundary condition, either constant or linear ramping
 *
 * @param _gravity initial gravity vector
 * @param _is_ramp whether the gravity is linear ramping or not
 * @param _ramp_step time when full gravity is reached
 * @param _gravity_end gravity value at end of ramp
 */
Gravity::Gravity(Vectorr _gravity,
        bool _is_ramp,
        int _ramp_step,
        Vectorr _gravity_end) {
  gravity = _gravity;
  is_ramp = _is_ramp;
  ramp_step = _ramp_step;
  gravity_end = _gravity_end;
}

struct CalculateGravity
  {
    Vectorr gravity;
    CalculateGravity(
        const Vectorr _gravity) : gravity(_gravity){};

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple tuple) const
    {

      Vectorr &force = thrust::get<0>(tuple);
      const Real mass = thrust::get<1>(tuple);

      if (mass <= 0.000000001) {return;}

      force += gravity * mass;
    }
  };

void Gravity::apply_on_nodes_f_ext(NodesContainer& nodes_ref) {

  const Real ramp_factor = ((Real)global_step_cpu)/ramp_step;
  if (is_ramp){

    if (global_step_cpu < ramp_step)
    {
      gravity = gravity_end*ramp_factor;
    }

  }
  execution_policy exec;
  PARALLEL_FOR_EACH_ZIP(exec,
                        nodes_ref.num_nodes_total,
                        CalculateGravity(gravity),
                        nodes_ref.forces_external_gpu.begin(),
                        nodes_ref.masses_gpu.begin());


  
};

}  // namespace pyroclastmpm