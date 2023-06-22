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

/**
 * @brief Creates a wall boundary condition on the nodes
 * @details The wall boundary condition is applied on the background grid
 *
 * The walls are defined by faces face0 =(x0, y0, z0) and face1 =(x1, y1, z1)
 *
 * \verbatim
 * In Two dimensions:
 *        x1
 *      +-----+
 *      |     |
 * y0   |     |  y1
 *      +-----+
 *        x0
 * \endverbatim
 *
 *
 * Modes can be applied to walls with face0_mode and face1_mode (0 - roller, 1 -
 * fixed)
 *
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/boundaryconditions/nodesdomain.h"
 *
 *        // set globals
 *
 *        // Create NodesContainer
 *
 *        // Only floor wall is roller but rest are fixed
 *        auto walls = NodeDomain(Vectori(0, 1, 0), Vectori(0.0, 0, 0));
 *
 *        // Add wall to Solver class
 *
 *  \endverbatim
 *
 */
class NodeDomain : public BoundaryCondition {

public:
  /// @brief Construct a new object
  /// @param face0_mode roller or fixed modes for cube face x0,y0,z0
  /// @param face1_mode roller or fixed modes  for cube face x1,y1,z1
  NodeDomain(Vectori _face0_mode = Vectori::Zero(),
             Vectori _face1_mode = Vectori::Zero());

  /// @brief Apply to node moments (walls)
  /// @param nodes_ref reference to NodesContainer
  void apply_on_nodes_moments(NodesContainer &nodes_ref,
                              ParticlesContainer &particles_ref) override;

  /// @brief roller or fixed modes (see class description)
  Vectori face1_mode;

  /// @brief roller or fixed modes (see class description)
  Vectori face0_mode;
};

} // namespace pyroclastmpm