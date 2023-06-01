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
  explicit USL(ParticlesContainer _particles, NodesContainer _nodes,
               cpu_array<MaterialType> _materials = cpu_array<MaterialType>(),
               cpu_array<BoundaryConditionType> _boundaryconditions =
                   cpu_array<BoundaryConditionType>(),
               Real _alpha = 0.99);

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
