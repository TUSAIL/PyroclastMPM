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

#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.h"
#include "pyroclastmpm/nodes/nodes.h"

namespace pyroclastmpm {

class NodesContainer;

struct Gravity : BoundaryCondition {
  // FUNCTIONS

  /**
   * @brief Gravity boundary condition, either constant or linear ramping
   *
   * @param _gravity initial gravity vector
   * @param _is_ramp whether the gravity is linear ramping or not
   * @param _ramp_step time when full gravity is reached
   * @param __gravity_end gravity value at end of ramp
   */
  Gravity(Vectorr _gravity, bool _is_ramp = false, int _ramp_step = 0,
          Vectorr _gravity_end = Vectorr::Zero());

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