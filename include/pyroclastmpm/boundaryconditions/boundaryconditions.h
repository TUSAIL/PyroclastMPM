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
 * @file boundaryconditions.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Boundary condition base class
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/nodes/nodes.h"
#include "pyroclastmpm/particles/particles.h"

namespace pyroclastmpm {

/// @brief The base class for every boundary condition
class BoundaryCondition {
public:
  /// @brief default constructor
  BoundaryCondition() = default;

  /// @brief Default destructor
  virtual ~BoundaryCondition() = default;

  /// @brief Apply to internal node forces
  /// @param nodes_ref reference to NodesContainer
  virtual void apply_on_nodes_loads(NodesContainer &nodes_ref){
      // override in derived classes
  };

  /// @brief Apply to node moments
  /// @param nodes_ref reference to NodesContainer
  virtual void apply_on_nodes_moments(NodesContainer &nodes_ref,
                                      ParticlesContainer &particles_ref){
      // override in derived classes
  };

  /// @brief Apply to external node forces
  /// @param nodes_ref reference to NodesContainer
  virtual void apply_on_nodes_f_ext(NodesContainer &nodes_ref){
      // override in derived classes
  };

  /// @brief Apply to particles
  /// @param nodes_ref reference to ParticlesContainer
  virtual void apply_on_particles(ParticlesContainer &particles_ref){
      // override in derived classes
  };

  /// @brief Output arrays stored in boundary conditions
  /// @param nodes_ref reference to NodesContainer
  /// @param particles_ref reference to ParticlesContainer
  virtual void output_vtk(NodesContainer &nodes_ref,
                          ParticlesContainer &particles_ref){
      // override in derived classes
  };

  virtual void initialize(const NodesContainer &nodes_ref,
                  const ParticlesContainer &particles_ref)
    {
        // override in derived classes
    };
                  

  /// @brief Flag if the boundary condition active or not
  bool isActive = true;
};

} // namespace pyroclastmpm