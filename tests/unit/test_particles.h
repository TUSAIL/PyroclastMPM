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

#include "pyroclastmpm/particles/particles.h"

// functions tested
// [x] ParticlesContainer::ParticlesContainer
// [x] ParticlesContainer::set_spatialpartition (implicitly calculate volume)
// [x] ParticlesContainer::partition (implicitly calculate volume)
// [x] ParticlesContainer::calculate_initial_volumes
// [x] ParticlesContainer::calculate_initial_masses
// [ ] ParticlesContainer::reset
// [ ] ParticlesContainer::reorder
// [ ] ParticlesContainer::output_vtk

using namespace pyroclastmpm;

TEST(ParticlesContainer, CalcVolumes) {

  Vectorr min = Vectorr::Zero();

  Vectorr max = Vectorr::Ones();

  Real cell_size = 0.2;

#if DIM == 3
  const std::vector<Vectorr> pos = {
      Vectorr({0.1, 0.1, 0.0}), Vectorr({0.199, 0.1, 0.0}),
      Vectorr({0.82, 0.0, 1.}), Vectorr({0.82, 0.6, 0.})};
#elif DIM == 2
  const std::vector<Vectorr> pos = {Vectorr({0.1, 0.1}), Vectorr({0.199, 0.1}),
                                    Vectorr({0.82, 0.0}), Vectorr({0.82, 0.6})};
#else
  const std::vector<Vectorr> pos = {Vectorr(0.1), Vectorr(0.199), Vectorr(0.82),
                                    Vectorr(0.6)};
#endif
  set_global_particles_per_cell(2);

  ParticlesContainer particles = ParticlesContainer(pos);

  particles.set_spatialpartition(min, max, cell_size);

  particles.calculate_initial_volumes();

  cpu_array<Real> volumes_cpu = particles.volumes_gpu;

#if DIM == 3
  const Real expected_volumes[4] = {0.004, 0.004, 0.004, 0.004};
  for (int pid = 0; pid < pos.size(); pid++) {
    EXPECT_NEAR(volumes_cpu[pid], expected_volumes[pid], 0.000001);
  }
#elif DIM == 2

  const Real expected_volumes[4] = {0.02, 0.02, 0.02, 0.02};
  for (int pid = 0; pid < pos.size(); pid++) {
    EXPECT_NEAR(volumes_cpu[pid], expected_volumes[pid], 0.000001);
  }
#else
  const Real expected_volumes[4] = {0.1, 0.1, 0.1, 0.1};
  for (int pid = 0; pid < pos.size(); pid++) {
    EXPECT_NEAR(volumes_cpu[pid], expected_volumes[pid], 0.000001);
  }
#endif
}

TEST(ParticlesContainer, CalculateMasses) {

  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real cell_size = 0.2;

#if DIM == 3
  const std::vector<Vectorr> pos = {
      Vectorr({0.1, 0.1, 0.0}), Vectorr({0.199, 0.1, 0.0}),
      Vectorr({0.82, 0.0, 1.}), Vectorr({0.82, 0.6, 0.})};
#elif DIM == 2
  const std::vector<Vectorr> pos = {Vectorr({0.1, 0.1}), Vectorr({0.199, 0.1}),
                                    Vectorr({0.82, 0.0}), Vectorr({0.82, 0.6})};
#else
  const std::vector<Vectorr> pos = {Vectorr(0.1), Vectorr(0.199), Vectorr(0.82),
                                    Vectorr(0.6)};
#endif

  set_global_particles_per_cell(2);

  ParticlesContainer particles = ParticlesContainer(pos);

  particles.set_spatialpartition(min, max, cell_size);

  particles.calculate_initial_volumes();

  particles.calculate_initial_masses(0, 0.5);

  cpu_array<Real> masses_cpu = particles.masses_gpu;

#if DIM == 3
  const Real expected_masses[4] = {0.002, 0.002, 0.002, 0.002};
  for (int pid = 0; pid < pos.size(); pid++) {
    EXPECT_NEAR(masses_cpu[pid], expected_masses[pid], 0.000001);
  }
#elif DIM == 2

  const Real expected_masses[4] = {0.01, 0.01, 0.01, 0.01};
  for (int pid = 0; pid < pos.size(); pid++) {
    EXPECT_NEAR(masses_cpu[pid], expected_masses[pid], 0.000001);
  }
#else
  const Real expected_masses[4] = {0.05, 0.05, 0.05, 0.05};
  for (int pid = 0; pid < pos.size(); pid++) {
    EXPECT_NEAR(masses_cpu[pid], expected_masses[pid], 0.000001);
  }
#endif
}