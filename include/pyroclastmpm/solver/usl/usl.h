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

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/solver/solver.h"
// #include "pyroclastmpm/solver/usl/usl_kernels.cuh"

namespace pyroclastmpm {

/**
 * @brief Update Stress Last solver (USL) class. This is the standard MPM
 * formulation with a mix of PIC/FLIP. Implementation based on the paper
 * de Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory,
 * implementation, and applications." Advances in applied mechanics 53 (2020):
 * 185-398. (Page 32)
 * TODO make alpha (FLIP/PIC ratio) a parameter (currently hardcoded)
 *
 */
struct USL : Solver {
public:
  // using Solver::Solver;
  /**
   * @brief Construct a new Solver object
   *
   * @param _particles particles container
   * @param _nodes nodes container
   * @param _boundaryconditions a list of boundary conditions to be applied
   * @param _materials a list of materials to be applied
   */
  explicit USL(
      const ParticlesContainer &_particles, const NodesContainer &_nodes,
      const cpu_array<MaterialType> &_materials = cpu_array<MaterialType>(),
      const cpu_array<BoundaryConditionType> &_boundaryconditions =
          cpu_array<BoundaryConditionType>(),
      Real _alpha = (Real)0.99);

  // FUNCTIONS
  /**
   * @brief Particle to grid update (velocity gather)
   *
   */
  void P2G();

  /**
   * @brief Grid to particle update (velocity scatter)
   *
   */
  void G2P();

  /*!
   * @brief  Solver one iteration of USL
   */
  void solve() override;

  /*!
   * @brief reset temporary arrays
   */
  void reset() override;

  /*!
   * @brief FLIP/PIC ratio
   */
  Real alpha;
};

} // namespace pyroclastmpm