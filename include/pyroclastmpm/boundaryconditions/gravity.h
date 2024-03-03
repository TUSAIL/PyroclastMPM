// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file gravity.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Gravity boundary conditions are applied to the background grid
 * via external forces on the nodes.
 *
 * @version 0.1
 * @date 2023-06-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.h"
#include "pyroclastmpm/nodes/nodes.h"

namespace pyroclastmpm
{

  /**
   * @brief Gravity boundary conditions
   * @details Gravity can either be constant
   * or have linear ramping to a final value.
   * \verbatim embed:rst:leading-asterisk
   *     Example usage (constant)
   *
   *     .. code-block:: cpp
   *
   *        #include "pyroclastmpm/boundaryconditions/gravity.h"
   *        #include "pyroclastmpm/nodes/nodes.h"
   *
   *        // set globals
   *
   *        // Create NodesContainer
   *
   *        auto gravity = Gravity(Vectorr(0.0, -9.81, 0.0));
   *
   *        // Add gravity to Solver class
   *
   * \endverbatim
   *
   * \verbatim embed:rst:leading-asterisk
   *    Example usage (ramping)
   *
   *   .. code-block:: cpp
   *
   *      auto gravity = Gravity(Vectorr(0.0, 0.0, 0.0),true,100,Vectorr(0.0,
   * -9.81, 0.0));
   *
   * \endverbatim
   *
   *
   */
  class Gravity : public BoundaryCondition
  {
  public:
    /// @brief Construct a new Gravity object
    /// @param _gravity gravity vector
    /// @param _is_ramp flag whether gravity is ramping linearly or not
    /// @param _ramp_step the amount of steps to ramp gravity to full value
    /// @param _gravity_end gravity value at end of ramp
    Gravity(Vectorr _gravity, bool _is_ramp = false, int _ramp_step = 0,
            Vectorr _gravity_end = Vectorr::Zero());

    /// @brief apply rigid body contact on background grid
    /// @param nodes_ref Nodes container
    /// @param particles_ref Particles container
    void apply_on_nodes_moments(NodesContainer &nodes_ref,
                                ParticlesContainer &particles_ref) override;

    /// @brief Initial gravity vector
    Vectorr gravity;

    /// @brief flag whether gravity is ramping linearly or not
    bool is_ramp;

    /// @brief the amount of steps to ramp gravity to full value
    int ramp_step;

    /// @brief gravity value at end of ramp
    Vectorr gravity_end;
  };

} // namespace pyroclastmpm