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
 * @file body_force.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief This file contains methods to apply a body force or moments nodes
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

namespace pyroclastmpm {

/**
 * @brief Body force boundary conditions
 * @details Applies a body force or moment on the background grid
 *
 * If the mode is "forces" then the body force is applied on the external
 * forces of the background grid
 *
 * If the mode is "moments" then moments are applied on the background grid,
 * or "fixed", meaning they are constraint to a fixed value
 *
 * \verbatim embed:rst:leading-asterisk
 *     Example usage (constant)
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/boundaryconditions/bodyforce.h"
 *        #include "pyroclastmpm/nodes/nodes.h"
 *
 *        // set globals
 *
 *        // Create NodesContainer with M number of nodes
 *
 *        /// NodesContainer give_node_coords() function can help to define
 * areas
 *        /// which are affected by the body force
 *
 *
 *        /// Mask of size M
 *        std::vector<bool> mask = {false, true,...};
 *
 *        std::vector<Vectorr> external_forces = {Vectorr({0., 0.25, 0.}),
 *                              Vectorr({0.8, 0.6, 0.4}),...};
 *
 *        /// Create BoundaryCondition
 *        auto boundarycondition = BodyForce("forces", values, mask);
 *
 *        // Add gravity to Solver class
 *
 * \endverbatim
 *
 *
 */
class BodyForce : public BoundaryCondition {

public:
  /// @brief Construct a new Body Force object
  /// @param _mode mode of the body force
  /// @param _values values of the body force
  /// @param _mask mask on which nodes to apply the body force
  BodyForce(const std::string_view &_mode, const cpu_array<Vectorr> &_values,
            const cpu_array<bool> &_mask) noexcept;

  /// @brief Apply body force to external node forces
  /// @param nodes_ptr NodeContainer reference
  void apply_on_nodes_f_ext(NodesContainer &nodes_ptr) override;

  /// @brief Update node values for moments
  /// @param nodes_ref NodeContainer reference
  /// @param particles_ref ParticleContainer reference
  void apply_on_nodes_moments(NodesContainer &nodes_ref,
                              ParticlesContainer &particles_ref) override;

  /**
   * @brief Values of the body force or moments
   * @details has the same size as the number of nodes
   */
  gpu_array<Vectorr> values_gpu;

  /**
   * @brief Mask on which nodes to apply the body force
   * @details has the same size as the number of nodes
   *
   */
  gpu_array<bool> mask_gpu;

  /**
   * @brief Mode of the body force
   * @details  0 forces, 1 is moments additve, 2 fixed moments
   */
  int mode_id;
};

} // namespace pyroclastmpm