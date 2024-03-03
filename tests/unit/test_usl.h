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

#include "pyroclastmpm/common/global_settings.h"
#include "pyroclastmpm/nodes/nodes.h"
#include "pyroclastmpm/particles/particles.h"
#include "pyroclastmpm/solver/usl/usl.h"

///  @brief Construct a new TEST object for ParticlesContainer G2P transfer
TEST(USL, Solve_StepWise)
{
  cpu_array<Matrix3r> stresses = std::vector({Matrix3r::Ones()});
#if DIM == 3
  cpu_array<Vectorr> pos =
      std::vector({Vectorr({0.1, 0.25, 0.3}), Vectorr({0.1, 0.25, 0.3})

      });
#elif DIM == 2
  cpu_array<Vectorr> pos =
      std::vector({Vectorr({0.1, 0.25}), Vectorr({0.1, 0.25})});
#else
  cpu_array<Vectorr> pos = std::vector({Vectorr(0.1), Vectorr(0.1)});
#endif

  cpu_array<Vectorr> vels = std::vector({Vectorr::Ones(), Vectorr::Ones()});

  cpu_array<Real> volumes = std::vector({0.7, 0.4});

  cpu_array<Real> masses = std::vector({0.1, 0.3});

  set_globals((Real)0.1, 1, "linear", "output");

  auto particles = pyroclastmpm::ParticlesContainer(pos, vels);

  auto nodes =
      pyroclastmpm::NodesContainer(Vectorr::Zero(), Vectorr::Ones(), 1.0);

  auto usl_solver = pyroclastmpm::USL(particles, nodes);

  auto mat = pyroclastmpm::LinearElastic(1000, (Real)0.1, (Real)0.1);

  usl_solver.materials.push_back(mat);

  usl_solver.particles.masses_gpu = masses;
  usl_solver.particles.volumes_gpu = volumes;
  usl_solver.particles.volumes_original_gpu = volumes;
  usl_solver.particles.stresses_gpu = stresses;
  usl_solver.particles.partition();

  calculate_shape_function(usl_solver.nodes, usl_solver.particles);

  usl_solver.P2G();

  cpu_array<Real> masses_cpu = usl_solver.nodes.masses_gpu;
  cpu_array<Vectorr> moments_cpu = usl_solver.nodes.moments_gpu;

#if DIM == 3
  const std::array<Real, 2> expected_node_mass = {0.189, 0.020999999};
  EXPECT_NEAR(masses_cpu[0], expected_node_mass[0], 0.000001);
  EXPECT_NEAR(masses_cpu[1], expected_node_mass[1], 0.000001);

  const std::array<Vectorr, 2> expected_node_moment = {
      Vectorr({0.189, 0.189, 0.189}),
      Vectorr({0.02099999, 0.02099999, 0.02099999})};
  EXPECT_NEAR(moments_cpu[0][0], expected_node_moment[0][0], 0.000001);
  EXPECT_NEAR(moments_cpu[0][1], expected_node_moment[0][1], 0.000001);
  EXPECT_NEAR(moments_cpu[0][2], expected_node_moment[0][2], 0.000001);
  EXPECT_NEAR(moments_cpu[1][0], expected_node_moment[1][0], 0.000001);
  EXPECT_NEAR(moments_cpu[1][1], expected_node_moment[1][1], 0.000001);
  EXPECT_NEAR(moments_cpu[1][2], expected_node_moment[1][2], 0.000001);
#elif DIM == 2
  const std::array<Real, 2> expected_node_mass = {0.27, 0.03};

  EXPECT_NEAR(masses_cpu[0], expected_node_mass[0], 0.000001);
  EXPECT_NEAR(masses_cpu[1], expected_node_mass[1], 0.000001);

  const std::array<Vectorr, 2> expected_node_moment = {Vectorr({0.27, 0.27}),
                                                       Vectorr({0.03, 0.03})};
  EXPECT_NEAR(moments_cpu[0][0], expected_node_moment[0][0], 0.000001);
  EXPECT_NEAR(moments_cpu[0][1], expected_node_moment[0][1], 0.000001);
  EXPECT_NEAR(moments_cpu[1][0], expected_node_moment[1][0], 0.000001);
  EXPECT_NEAR(moments_cpu[1][1], expected_node_moment[1][1], 0.000001);
#else
  const std::array<Real, 2> expected_node_mass = {0.36, 0.04};
  EXPECT_NEAR(masses_cpu[0], expected_node_mass[0], 0.000001);
  EXPECT_NEAR(masses_cpu[1], expected_node_mass[1], 0.000001);

  const std::array<Vectorr, 2> expected_node_moment = {Vectorr(0.36),
                                                       Vectorr(0.04)};
  EXPECT_NEAR(moments_cpu[0][0], expected_node_moment[0][0], 0.000001);
  EXPECT_NEAR(moments_cpu[1][0], expected_node_moment[1][0], 0.000001);
#endif

  usl_solver.G2P();

  cpu_array<Real> volumes_cpu = usl_solver.particles.volumes_gpu;
  cpu_array<Vectorr> positions_cpu = usl_solver.particles.positions_gpu;
  cpu_array<Vectorr> velocities_cpu = usl_solver.particles.velocities_gpu;
  cpu_array<Matrixr> F_cpu = usl_solver.particles.F_gpu;

  // we only test one particle here
#if DIM == 3

  const Real expected_volume = 0.4402222222222;

  const Vectorr expected_position = Vectorr({0.2, 0.35, 0.4});
  const Vectorr expected_velocities = Vectorr({1.0, 1.0, 1.0});

  // TODO tests non diagonal F
  const Matrixr expected_F = Matrixr(
      {{0.805555555, 0.0, 0.0}, {0.0, 0.90666666, 0.0}, {0.0, 0.0, 0.916666}});

  EXPECT_NEAR(volumes_cpu[0], expected_volume, 0.000001);
  EXPECT_NEAR(positions_cpu[0][0], expected_position[0], 0.000001);
  EXPECT_NEAR(positions_cpu[0][1], expected_position[1], 0.000001);
  EXPECT_NEAR(positions_cpu[0][2], expected_position[2], 0.000001);
  EXPECT_NEAR(velocities_cpu[0][0], expected_velocities[0], 0.000001);
  EXPECT_NEAR(velocities_cpu[0][1], expected_velocities[1], 0.000001);
  EXPECT_NEAR(velocities_cpu[0][1], expected_velocities[1], 0.000001);
  EXPECT_NEAR(F_cpu[0](0), expected_F(0), 0.000001);
  EXPECT_NEAR(F_cpu[0](4), expected_F(4), 0.000001);
  EXPECT_NEAR(F_cpu[0](8), expected_F(8), 0.000001);

#elif DIM == 2

  const Real expected_volume = 0.49855555;

  const auto expected_position = Vectorr({0.2, 0.35});
  const auto expected_velocities = Vectorr({1.0, 1.0});

  // TODO tests non diagonal F
  const auto expected_F = Matrixr({{0.805555555, 0.0}, {0.0, 0.90666666}});

  EXPECT_NEAR(volumes_cpu[0], expected_volume, 0.000001);
  EXPECT_NEAR(positions_cpu[0][0], expected_position[0], 0.000001);
  EXPECT_NEAR(positions_cpu[0][1], expected_position[1], 0.000001);
  EXPECT_NEAR(velocities_cpu[0][0], expected_velocities[0], 0.000001);
  EXPECT_NEAR(velocities_cpu[0][1], expected_velocities[1], 0.000001);
  EXPECT_NEAR(F_cpu[0](0), expected_F(0), 0.000001);
  EXPECT_NEAR(F_cpu[0](3), expected_F(3), 0.000001);
#else
  const Real expected_volume = 0.563888888;

  const Vectorr expected_position = Vectorr(0.2);
  const Vectorr expected_velocities = Vectorr(1.0);
  const Matrixr expected_F = Matrixr(0.805555555);

  EXPECT_NEAR(volumes_cpu[0], expected_volume, 0.000001);
  EXPECT_NEAR(positions_cpu[0][0], expected_position[0], 0.000001);
  EXPECT_NEAR(velocities_cpu[0][0], expected_velocities[0], 0.000001);
  EXPECT_NEAR(F_cpu[0][0], expected_F[0], 0.000001);

#endif
}

TEST(USL, Solve)
{
  // Check if full solve() works
#if DIM == 3
  cpu_array<Vectorr> pos =
      std::vector({Vectorr({0.1, 0.25, 0.3}), Vectorr({0.1, 0.25, 0.3})});
#elif DIM == 2
  cpu_array<Vectorr> pos =
      std::vector({Vectorr({0.1, 0.25}), Vectorr({0.1, 0.25})});
#else
  cpu_array<Vectorr> pos = std::vector({Vectorr(0.1), Vectorr(0.1)});
#endif

  cpu_array<Vectorr> vels = std::vector({Vectorr::Ones(), Vectorr::Ones()});

  set_globals((Real)0.1, 1, "linear", "output");

  auto particles = pyroclastmpm::ParticlesContainer(pos, vels);

  auto nodes =
      pyroclastmpm::NodesContainer(Vectorr::Zero(), Vectorr::Ones(), 1.0);

  auto usl_solver = pyroclastmpm::USL(particles, nodes);

  auto mat = pyroclastmpm::LinearElastic(1000, (Real)0.1, (Real)0.1);

  auto bc = pyroclastmpm::BoundaryCondition();

  usl_solver.materials.push_back(mat);

  usl_solver.boundaryconditions.push_back(bc);

  // internal functions are tested individually
  usl_solver.reset();

  // internal functions are tested individually
  usl_solver.solve();

  // check if it runs without errors
}