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

#include "pyroclastmpm/materials/linearelastic.h"

#include "pyroclastmpm/particles/particles.h"

#include "pyroclastmpm/nodes/nodes.h"

#include "pyroclastmpm/solver/solver.h"

// features to test
// [x] Solver::Solver
// [x] Solver::stress_update (implicitly tested by material)
// [x] Solver::solve_nsteps (implicitly tested)
// [x] Solver::calculate_shape_function (implicitly tested by shapefunctions)

/**
 * @brief Construct a new TEST object to test if solver constructor works
 * Only tests if the constructor works, not particle and node initialization
 * functions
 *
 */
TEST(Solver, Constructor) {
  set_globals(0.1, 1, pyroclastmpm::LinearShapeFunction, "output");
  std::vector<Vectorr> pos = {Vectorr::Zero()};

  std::vector<pyroclastmpm::Material> materials;

  materials.push_back(LinearElastic(1000, 10, 10));

  auto particles = pyroclastmpm::ParticlesContainer(pos);

  pyroclastmpm::NodesContainer nodes(Vectorr::Zero(), Vectorr::Ones(), 0.5);

  auto solver = pyroclastmpm::Solver(particles, nodes, materials);
};