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

TEST(LinearElastic, StressUpdateLinearElastic) {

#if DIM == 3
  const std::vector<Vectorr> pos = {Vectorr({0.1, 0.1, 0.0})};
#elif DIM == 2
  const std::vector<Vectorr> pos = {Vectorr({0.1, 0.1})};
#else
  const std::vector<Vectorr> pos = {Vectorr(0.1)};
#endif
  Real dt = 0.1;
  set_global_dt(dt);
  const auto velgrad = Matrixr::Identity() * 0.1;
  ParticlesContainer particles = ParticlesContainer(pos);

  LinearElastic mat = LinearElastic(1000, 0.1, 0.1);

  const std::vector<Matrixr> F = {(Matrixr::Identity() + velgrad * dt) *
                                  Matrixr::Identity()};
  particles.F_gpu = F;

  mat.stress_update(particles, 0);

  cpu_array<Matrix3r> stresses = particles.stresses_gpu;

#if DIM == 3
  Matrix3r expected_stress;
  expected_stress << 0.001023, 0, 0, 0, 0.001023, 0, 0, 0, 0.001023;
  EXPECT_NEAR(stresses[0](0), expected_stress(0), 0.0001);
  EXPECT_NEAR(stresses[0](4), expected_stress(4), 0.0001);
  EXPECT_NEAR(stresses[0](8), expected_stress(8), 0.0001);
#elif DIM == 2
  Matrix3r expected_stress;
  expected_stress << 0.001023, 0, 0, 0, 0.001023, 0, 0, 0, 0;
  EXPECT_NEAR(stresses[0](0), expected_stress(0), 0.0001);
  EXPECT_NEAR(stresses[0](4), expected_stress(4), 0.0001);
#else
  Matrix3r expected_stress;
  expected_stress << 0.001023, 0, 0, 0, 0, 0, 0, 0, 0;
  EXPECT_NEAR(stresses[0](0), expected_stress(0), 0.0001);
#endif
}