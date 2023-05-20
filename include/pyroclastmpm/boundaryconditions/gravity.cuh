#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"

namespace pyroclastmpm
{

  class NodesContainer;

  struct Gravity : BoundaryCondition
  {
    // FUNCTIONS

    /**
     * @brief Gravity boundary condition, either constant or linear ramping
     *
     * @param _gravity initial gravity vector
     * @param _is_ramp whether the gravity is linear ramping or not
     * @param _ramp_step time when full gravity is reached
     * @param __gravity_end gravity value at end of ramp
     */
    Gravity(Vectorr _gravity, bool _is_ramp = false, int _ramp_step = 0, Vectorr _gravity_end = Vectorr::Zero());

    ~Gravity(){};
    void apply_on_nodes_f_ext(NodesContainer &nodes_ptr) override;

    /**
     * @brief Initial gravity vector
     *
     */
    Vectorr gravity;

    /**
     * @brief flag whether gravity is ramping linearly or not
     *
     */
    bool is_ramp;

    /**
     * @brief the amount of steps to ramp gravity to full value
     *
     */
    int ramp_step;

    /**
     * @brief gravity value at end of ramp
     *
     */
    Vectorr gravity_end;
  };

} // namespace pyroclastmpm