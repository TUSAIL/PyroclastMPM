#include "pyroclastmpm/boundaryconditions/gravity/gravity.cuh"

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

void Gravity::apply_on_nodes_f_ext(NodesContainer& nodes_ref) {

  const Real ramp_factor = ((Real)global_step_cpu)/ramp_step;
  if (is_ramp){

    if (global_step_cpu < ramp_step)
    {
      gravity = gravity_end*ramp_factor;
    }

  }

  KERNEL_APPLY_GRAVITY<<<nodes_ref.launch_config.tpb,
                         nodes_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(nodes_ref.forces_external_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()), gravity,
      nodes_ref.num_nodes_total);
  gpuErrchk(cudaDeviceSynchronize());
  
};

}  // namespace pyroclastmpm