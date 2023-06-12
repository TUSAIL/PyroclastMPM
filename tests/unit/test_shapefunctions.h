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

#include "pyroclastmpm/shapefunction/shapefunction.h"

// Functions to test
// [x] LinearShapeFunction
// [ ] QuadraticShapeFunction (to be fixed)
// [x] CubicShapeFunction

using namespace pyroclastmpm;

TEST(ShapeFunctions, LinearShapeFunction) {
  set_global_shapefunction(LinearShapeFunction);
#if DIM == 3
  std::vector<Vectorr> pos = {Vectorr({0.45, 0.21, 0.1})};
#elif DIM == 2
  std::vector<Vectorr> pos = {Vectorr({0.45, 0.21})};
#else
  std::vector<Vectorr> pos = {Vectorr(0.94)};
#endif

  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real nodal_spacing = 0.1;

  NodesContainer nodes = NodesContainer(min, max, nodal_spacing);

  ParticlesContainer particles = ParticlesContainer(pos);

  particles.set_spatialpartition(nodes.node_start, nodes.node_end,
                                 nodes.node_spacing);

  calculate_shape_function(nodes, particles);

  cpu_array<Real> psi = particles.psi_gpu;
  cpu_array<Vectorr> dpsi = particles.dpsi_gpu;

  Real sum_psi = 0.0;
  for (int i = 0; i < psi.size(); i++) {
    sum_psi += psi[i];
  }
  EXPECT_NEAR(sum_psi, 1., 0.0001);

#if DIM == 3

  EXPECT_NEAR(psi[0], 0.45, 0.0001);
  EXPECT_NEAR(psi[1], 0., 0.0001);
  EXPECT_NEAR(psi[2], 0.45, 0.0001);
  EXPECT_NEAR(psi[3], 0.0, 0.0001);
  EXPECT_NEAR(psi[4], 0.05, 0.0001);
  EXPECT_NEAR(psi[5], 0.0, 0.0001);
  EXPECT_NEAR(psi[6], 0.05, 0.0001);
  EXPECT_NEAR(psi[7], 0.0, 0.0001);

  EXPECT_NEAR(dpsi[0][0], -9, 0.0001);
  EXPECT_NEAR(dpsi[0][1], -5, 0.0001);
  EXPECT_NEAR(dpsi[0][2], 0., 0.0001);

  EXPECT_NEAR(dpsi[1][0], 0, 0.0001);
  EXPECT_NEAR(dpsi[1][1], 0, 0.0001);
  EXPECT_NEAR(dpsi[1][2], 4.5, 0.0001);

  EXPECT_NEAR(dpsi[2][0], 9, 0.0001);
  EXPECT_NEAR(dpsi[2][1], -5, 0.0001);
  EXPECT_NEAR(dpsi[2][2], 0, 0.0001);
  EXPECT_NEAR(dpsi[3][0], 0, 0.0001);
  EXPECT_NEAR(dpsi[3][1], 0, 0.0001);
  EXPECT_NEAR(dpsi[3][2], 4.5, 0.0001);
  EXPECT_NEAR(dpsi[4][0], -1, 0.0001);
  EXPECT_NEAR(dpsi[4][1], 5, 0.0001);
  EXPECT_NEAR(dpsi[4][2], 0, 0.0001);
  EXPECT_NEAR(dpsi[5][0], 0, 0.0001);
  EXPECT_NEAR(dpsi[5][1], 0, 0.0001);
  EXPECT_NEAR(dpsi[5][2], 0.5, 0.0001);

#elif DIM == 2
  EXPECT_NEAR(psi[0], 0.45, 0.0001);
  EXPECT_NEAR(psi[1], 0.45, 0.0001);
  EXPECT_NEAR(psi[2], 0.05, 0.0001);
  EXPECT_NEAR(psi[3], 0.05, 0.0001);

  EXPECT_NEAR(dpsi[0][0], -9, 0.0001);
  EXPECT_NEAR(dpsi[0][1], -5, 0.0001);
  EXPECT_NEAR(dpsi[1][0], 9, 0.0001);
  EXPECT_NEAR(dpsi[1][1], -5, 0.0001);
  EXPECT_NEAR(dpsi[2][0], -1, 0.0001);
  EXPECT_NEAR(dpsi[2][1], 5, 0.0001);
  EXPECT_NEAR(dpsi[3][0], 1, 0.0001);
  EXPECT_NEAR(dpsi[3][1], 5, 0.0001);

#else
  EXPECT_NEAR(psi[0], 0.6, 0.0001);
  EXPECT_NEAR(psi[1], 0.4, 0.0001);

  EXPECT_NEAR(dpsi[0][0], -10, 0.0001);
  EXPECT_NEAR(dpsi[1][0], 10, 0.0001);

#endif
}

TEST(ShapeFunctions, CubicShapeFunction) {
  set_global_shapefunction(CubicShapeFunction);
#if DIM == 3
  std::vector<Vectorr> pos = {Vectorr({0.45, 0.21, 0.1})};
#elif DIM == 2
  std::vector<Vectorr> pos = {Vectorr({0.45, 0.21})};
#else
  std::vector<Vectorr> pos = {Vectorr(0.45)};
#endif

  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real nodal_spacing = 0.1;

  NodesContainer nodes = NodesContainer(min, max, nodal_spacing);
  ParticlesContainer particles = ParticlesContainer(pos);

  particles.set_spatialpartition(nodes.node_start, nodes.node_end,
                                 nodes.node_spacing);

  calculate_shape_function(nodes, particles);

  cpu_array<Real> psi = particles.psi_gpu;

  cpu_array<Vectorr> dpsi = particles.dpsi_gpu;

  Real sum_psi = 0.0;
  for (int i = 0; i < psi.size(); i++) {
    sum_psi += psi[i];
  }

  EXPECT_NEAR(sum_psi, 1., 0.0001);

  // we only check some nodes
  // should already give us an idea if the function is correct
#if DIM == 3
  EXPECT_NEAR(psi[0], 0.00042187, 0.0001);
  EXPECT_NEAR(psi[1], 0.00168750, 0.0001);
  EXPECT_NEAR(psi[2], 0.00042187, 0.0001);
  EXPECT_NEAR(psi[3], 0, 0.0001);
  EXPECT_NEAR(psi[4], 0.00970313, 0.0001);
  EXPECT_NEAR(psi[5], 0.038812, 0.0001);
  EXPECT_NEAR(psi[6], 0.00970313, 0.0001);
  EXPECT_NEAR(psi[7], 0., 0.0001);

  EXPECT_NEAR(dpsi[0][0], -0.02531, 0.0001);
  EXPECT_NEAR(dpsi[0][1], -0.01406, 0.0001);
  EXPECT_NEAR(dpsi[0][2], -0.01266, 0.0001);

  EXPECT_NEAR(dpsi[1][0], -0.1013, 0.0001);
  EXPECT_NEAR(dpsi[1][1], -0.05625, 0.0001);
  EXPECT_NEAR(dpsi[1][2], 0, 0.0001);

  EXPECT_NEAR(dpsi[2][0], -0.02531, 0.0001);
  EXPECT_NEAR(dpsi[2][1], -0.01406, 0.0001);
  EXPECT_NEAR(dpsi[2][2], 0.01266, 0.0001);

  EXPECT_NEAR(dpsi[3][0], 0, 0.0001);
  EXPECT_NEAR(dpsi[3][1], 0, 0.0001);
  EXPECT_NEAR(dpsi[3][2], 0, 0.0001);

  EXPECT_NEAR(dpsi[4][0], -0.1266, 0.0001);
  EXPECT_NEAR(dpsi[4][1], -0.3234, 0.0001);
  EXPECT_NEAR(dpsi[4][2], -0.2911, 0.0001);

  EXPECT_NEAR(dpsi[5][0], -0.5063, 0.0001);
  EXPECT_NEAR(dpsi[5][1], -1.29375, 0.0001);
  EXPECT_NEAR(dpsi[5][2], 0, 0.0001);

#elif DIM == 2
  EXPECT_NEAR(psi[0], 0.0025312525, 0.0001);
  EXPECT_NEAR(psi[1], 0.0582187771, 0.0001);
  EXPECT_NEAR(psi[2], 0.0582187585, 0.0001);
  EXPECT_NEAR(psi[3], 0.002531249076128006, .0001);

  EXPECT_NEAR(dpsi[0][0], -0.1518751, 0.0001);
  EXPECT_NEAR(dpsi[0][1], -0.084375, 0.0001);
  EXPECT_NEAR(dpsi[1][0], -0.7593751, 0.0001);
  EXPECT_NEAR(dpsi[1][1], -1.94062566, 0.0001);
  EXPECT_NEAR(dpsi[2][0], 0.759375274, 0.0001);
  EXPECT_NEAR(dpsi[2][1], -1.94062507, 0.0001);
  EXPECT_NEAR(dpsi[3][0], 0.15187497, 0.0001);
  EXPECT_NEAR(dpsi[3][1], -0.08437495, 0.0001);

#else
  EXPECT_NEAR(psi[0], 0.02083333, 0.0001);
  EXPECT_NEAR(psi[1], 0.47916666, 0.0001);

  EXPECT_NEAR(dpsi[0][0], -1.25, 0.0001);
  EXPECT_NEAR(dpsi[1][0], -6.25, 0.0001);
#endif
}