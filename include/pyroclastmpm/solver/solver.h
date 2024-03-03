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
 * @file solver.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Solver base class combines all the components of the MPM solver
 * @details The solver is responsible for the main loop of the simulation
 * The base class is responsible for defining the initialization, output,
 * and stress update procedures. These procedures are further called or
 * modified by the derived classes.
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */
#pragma once

// TODO: remove this if not used
// #include <indicators/progress_bar.hpp>
// #include <indicators/cursor_control.hpp>

#include <variant>

// Common
#include "pyroclastmpm/common/global_settings.h"
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
#include "pyroclastmpm/materials/mcc_mu_i.h"
#include "pyroclastmpm/materials/modifiedcamclay.h"
#include "pyroclastmpm/materials/muijop.h"
#include "pyroclastmpm/materials/newtonfluid.h"

// Particles, Nodes and shapefunctions
#include "pyroclastmpm/nodes/nodes.h"
#include "pyroclastmpm/particles/particles.h"
#include "pyroclastmpm/shapefunction/shapefunction.h"

namespace pyroclastmpm {

/// @brief All the possible materials used in the simulation
using MaterialType =
    std::variant<Material, LinearElastic, NewtonFluid, LocalGranularRheology,
                 ModifiedCamClay, MuIJop, MCCMuI>;

/// @brief All possible boundary conditions used in the simulation
using BoundaryConditionType =
    std::variant<BoundaryCondition, Gravity, RigidBodyLevelSet, BodyForce,
                 PlanarDomain, NodeDomain>;

/**
 * @brief Solver base class
 * @details The solver is responsible for the main loop of the simulation
 * The base class is responsible for defining the initialization, output,
 * and stress update procedures. These procedures are further called or
 * modified by the derived classes.
 *
 *
 */
class Solver {
public:
  /// @brief Construct a new Solver object
  /// @param _particles A ParticlesContainer class
  /// @param _nodes A NodesContainer class
  /// @param _boundaryconditions A list of boundary conditions to be applied
  /// @param _materials A list of materials to be applied
  explicit Solver(
      const ParticlesContainer &_particles, const NodesContainer &_nodes,
      const cpu_array<MaterialType> &_materials = cpu_array<MaterialType>(),
      const cpu_array<BoundaryConditionType> &_boundaryconditions =
          cpu_array<BoundaryConditionType>());

  /// @brief Destroy the Solver object
  virtual ~Solver();

  /// @brief Solve the main loop for n_steps
  /// @param n_steps
  void solve_nsteps(int n_steps);

  /// @brief Output the results (ParticlesContainer,NodesContainer, etc. )
  /// @details calls .output_vtk() for all the components
  void output();

  /// @brief Do stress update for all the materials
  void stress_update();

  ///@brief reset (temporary) arrays to initial state
  ///@details override this function in derived classes
  virtual void reset(){
      /// override this function in derived classes
  };

  ///@brief main loop of the solver
  ///@details override this function in derived classes
  virtual void solve(){
      // override this function in derived classes
  };

  /// @brief Solve the main loop for n_steps
  /// @param total_steps number of steps to solve for
  /// @param output_frequency output frequency
  void run(const int total_steps, const int output_frequency);

  ///@brief NodesContainer
  NodesContainer nodes;

  ///@brief ParticlesContainer
  ParticlesContainer particles;

  ///@brief A list of MaterialType
  cpu_array<MaterialType> materials;

  ///@brief A list of BoundaryConditionType
  cpu_array<BoundaryConditionType> boundaryconditions;

  /** current step of the main loop */
  int current_step;
  

 /// @brief Total memory 
  double total_memory_mb = 0.0;
};

} // namespace pyroclastmpm