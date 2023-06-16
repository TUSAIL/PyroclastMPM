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

#include <variant>

// Common
#include "pyroclastmpm/common/types_common.h"

// Boundary conditions
#include "pyroclastmpm/boundaryconditions/bodyforce.h"
#include "pyroclastmpm/boundaryconditions/boundaryconditions.h"
#include "pyroclastmpm/boundaryconditions/gravity.h"
#include "pyroclastmpm/boundaryconditions/nodedomain.h"
#include "pyroclastmpm/boundaryconditions/planardomain.h"
#include "pyroclastmpm/boundaryconditions/rigidbodylevelset.h"

// Materials
#include "pyroclastmpm/materials/linearelastic.h"
#include "pyroclastmpm/materials/localrheo.h"
#include "pyroclastmpm/materials/materials.h"
#include "pyroclastmpm/materials/newtonfluid.h"

// Particles, Nodes and shapefunctions
#include "pyroclastmpm/nodes/nodes.h"
#include "pyroclastmpm/particles/particles.h"
#include "pyroclastmpm/shapefunction/shapefunction.h"

namespace pyroclastmpm {

/**
 * @brief Define the material type as a variant of all the possible materials
 *
 */
using MaterialType =
    std::variant<Material, LinearElastic, NewtonFluid, LocalGranularRheology>;

/**
 * @brief Define the boundary condition type as a variant of all the possible
 * boundary conditions
 *
 */
using BoundaryConditionType =
    std::variant<BoundaryCondition, Gravity, RigidBodyLevelSet, BodyForce,
                 PlanarDomain, NodeDomain>;

/**
 * @brief MPM solver base class
 *
 */
class Solver {
public:
  /**
   * @brief Construct a new Solver object
   *
   * @param _particles particles container
   * @param _nodes nodes container
   * @param _boundaryconditions a list of boundary conditions to be applied
   * @param _materials a list of materials to be applied
   */
  explicit Solver(
      const ParticlesContainer &_particles, const NodesContainer &_nodes,
      const cpu_array<MaterialType> &_materials = cpu_array<MaterialType>(),
      const cpu_array<BoundaryConditionType> &_boundaryconditions =
          cpu_array<BoundaryConditionType>());

  /**
   * @brief Destroy the Solver object
   *
   */
  virtual ~Solver();

  /**
   * @brief Solve the main loop for n_steps
   *
   * @param n_steps
   */
  void solve_nsteps(int n_steps);

  /**
   * @brief Output the results (particles,nodes,boundaryconditions, etc. )
   *
   */
  void output();

  /**
   * @brief Do stress update for all particles (using constitutive law)
   *
   */
  void stress_update();

  /**
   * @brief reset (temporary) arrays to initial state
   *
   */
  virtual void reset(){};

  /**
   * @brief main loop of the solver
   *
   */
  virtual void solve(){};

  /*!
   * @brief Smart pointer to nodes container
   */
  NodesContainer nodes;

  /*!
   * @brief Smart pointer to particles container
   */
  ParticlesContainer particles;

  /*! @brief list of materials */
  cpu_array<MaterialType> materials;

  /*!
   * @brief a list of pointers to the boundary conditions
   */
  cpu_array<BoundaryConditionType> boundaryconditions;

  /** current step of the main loop */
  int current_step;
};

} // namespace pyroclastmpm