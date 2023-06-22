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
 * @file usl.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Update Stress Last (USL) solver
 * @details This is the most common MPM solver.
 * The implementation is based on the paper
 * de Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory,
 * implementation, and applications." Advances in applied mechanics 53 (2020):
 * 185-398. (Page 32)
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */
#pragma once

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/solver/solver.h"

namespace pyroclastmpm {

/**
 * @brief Update Stress Last (USL) Solver class.
 * @details This standard MPM has mixed PIC/FLIP integration schemes.
 * The implementation is based on the paper
 * de Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory,
 * implementation, and applications." Advances in applied mechanics 53 (2020):
 * 185-398. (Page 32)
 * It is important that global variables are set before the solver is
 * called. This can be done by calling the set_globals() function.
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *      Solving one iteration of the USL solver with a linear elastic material
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/nodes/nodes.h"
 *        #include "pyroclastmpm/particles/particles.h"
 *        #include "pyroclastmpm/solver/usl/usl.h"
 *
 *        set_globals(((Real) 0.1 , 1, "linear", "output");
 *
 *        ...
 *
 *        auto nodes = NodesContainer(min, max, nodal_spacing);
 *        auto particles = ParticlesContainer(pos, vels);
 *        auto mat = LinearElastic(1000, 0.1, 0.1);
 *
 *        auto usl_solver = USL(particles, nodes, {mat});
 *
 *        for (int i = 0; i < 5; i++)
 *        {
 *          usl_solver.solve();
 *        }
 *
 * \endverbatim
 *
 */
class USL : public Solver {
public:
  ///@brief Construct a new USL object
  ///@param _particles ParticlesContainer class
  ///@param _nodes NodesContainer class
  ///@param _materials A list of Materials
  ///@param _boundaryconditions a list of boundary conditions
  ///@param _alpha Flip/PIC mixture
  explicit USL(
      const ParticlesContainer &_particles, const NodesContainer &_nodes,
      const cpu_array<MaterialType> &_materials = cpu_array<MaterialType>(),
      const cpu_array<BoundaryConditionType> &_boundaryconditions =
          cpu_array<BoundaryConditionType>(),
      Real _alpha = (Real)0.99);

  /// @brief Particle to grid update (gather)
  void P2G();

  ///  @brief Grid to particle update (scatter)
  void G2P();

  /// @brief  Solver one iteration of USL
  void solve() override;

  /// @brief reset temporary arrays
  void reset() override;

  /// @brief FLIP/PIC ratio
  Real alpha;
};

} // namespace pyroclastmpm