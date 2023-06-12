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

// #include "pyroclastmpm/nodes/nodes.cuh"

namespace pyroclastmpm {

/**
 * @brief Apply a domain to the simulation
 *
 */
struct PlanarDomain : BoundaryCondition {

  // FUNCTIONS
  /**
   * @brief Construct a new Pinball domain object
   *
   * @param axis0_friction friction of the x0,y0,z0 axes
   * @param axis1_friction friction of the x1,y1,z1 axes
   */
  PlanarDomain(Vectorr _axis0_friction = Vectorr::Zero(),
               Vectorr _axis1_friction = Vectorr::Zero());

  ~PlanarDomain(){};

  Vectorr axis0_friction;

  Vectorr axis1_friction;
  /**
   * @brief Apply boundary conditions on node moments
   *
   * @param nodes_ptr NodesContainer object
   */
  void apply_on_particles(ParticlesContainer &particles_ref) override;
};

} // namespace pyroclastmpm