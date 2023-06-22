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

namespace pyroclastmpm {

/**
 * @brief Plane boundary condition on the nodes
 * @details DEM style boundary condition where contact between particles and
 * the plane is modelled with a frictional contact model.
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
 * Friction is applied to walls with face0_friction and face1_friction in
 * degrees
 *
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/boundaryconditions/planardomain.h"
 *
 *        // set globals
 *
 *        // floor with friction of 15 and rest no friction
 *        auto walls = PlanarDomain(Vectori(0, 15, 0), Vectori(0.0, 0, 0));
 *
 *        // Add wall to Solver class
 *
 *  \endverbatim
 *
 */
class PlanarDomain : public BoundaryCondition {
public:
  /// @brief Construct a new object
  /// @param face0_friction Friction angle (degrees) for cube face x0,y0,z0
  /// @param face1_friction Friction angle (degrees)for cube face x1,y1,z1
  PlanarDomain(Vectorr _face0_friction = Vectorr::Zero(),
               Vectorr _face1_friction = Vectorr::Zero());

  /// @brief face0_friction Friction angle (degrees) for cube face x0,y0,z0
  Vectorr face0_friction;

  /// @brief face1_friction Friction angle degrees for cube face x0,y0,z0
  Vectorr face1_friction;

  /// @brief apply contact on particles
  /// @param particles_ref ParticlesContainer reference
  void apply_on_particles(ParticlesContainer &particles_ref) override;
};

} // namespace pyroclastmpm