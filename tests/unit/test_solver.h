
#pragma once

#include "pyroclastmpm/materials/linearelastic.h"

#include "pyroclastmpm/particles/particles.h"

#include "pyroclastmpm/nodes/nodes.h"

#include "pyroclastmpm/solver/solver.h"

// features to test
// [x] Solver::Solver
// [x] Solver::stress_update (implicitly tested by material)
// [x] Solver::solve_nsteps (implicitly tested)
// [x] Solver::calculate_shape_function (implcitly tested by shapefunctions)

using namespace pyroclastmpm;

/**
 * @brief Construct a new TEST object to test if solver constructor works
 * Only tests if the constructor works, not particle and node initialization
 * functions
 *
 */
TEST(Solver, Constructor) {
  set_globals(0.1, 1, LinearShapeFunction, "output");
  std::vector<Vectorr> pos = {Vectorr::Zero()};

  std::vector<Material> materials = {LinearElastic(1000, 10, 10)};

  ParticlesContainer particles = ParticlesContainer(pos);

  NodesContainer nodes(Vectorr::Zero(), Vectorr::Ones(), 0.5);

  Solver solver = Solver(particles, nodes, materials);
};