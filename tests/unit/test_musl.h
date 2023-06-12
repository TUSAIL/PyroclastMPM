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
#include "pyroclastmpm/solver/musl/musl.h"

// Features to be tested
// [x] MUSL::USL (implicitly through solver, inherits)
// [x] MUSL::solve (implicitly through all other tests)
// [x] MUSL::G2P_doublemapping
// [x] MUSL::P2G_doublemapping
// [x] MUSL::G2P

using namespace pyroclastmpm;

/**
 * @brief Construct a new TEST object for ParticlesContainer G2P transfer
 *
 */
TEST(MUSL, Solve)

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

  set_globals(0.1, 1, LinearShapeFunction, "output");

  ParticlesContainer particles = ParticlesContainer(pos, vels);

  NodesContainer nodes = NodesContainer(Vectorr::Zero(), Vectorr::Ones(), 1.0);

  MUSL musl_solver = MUSL(particles, nodes);

  LinearElastic mat = LinearElastic(1000, 0.1, 0.1);
  musl_solver.materials.push_back(mat);

  musl_solver.particles.masses_gpu = masses;
  musl_solver.particles.volumes_gpu = volumes;
  musl_solver.particles.volumes_original_gpu = volumes;
  musl_solver.particles.stresses_gpu = stresses;
  musl_solver.particles.partition();

  musl_solver.calculate_shape_function();

  musl_solver.P2G();

  musl_solver.nodes.integrate();

  musl_solver.G2P_double_mapping();

  cpu_array<Vectorr> particle_vels_cpu = musl_solver.particles.velocities_gpu;
  cpu_array<Vectorr> particle_pos_cpu = musl_solver.particles.positions_gpu;

#if DIM == 3
  EXPECT_NEAR(particle_vels_cpu[0][0], 1.0, 0.000001);
  EXPECT_NEAR(particle_vels_cpu[0][1], 1.0, 0.000001);
  EXPECT_NEAR(particle_vels_cpu[0][2], 1.0, 0.000001);

  EXPECT_NEAR(particle_pos_cpu[0][0], 0.2, 0.000001);
  EXPECT_NEAR(particle_pos_cpu[0][1], 0.3500, 0.000001);
  EXPECT_NEAR(particle_pos_cpu[0][2], 0.40, 0.000001);
#elif DIM == 2

  EXPECT_NEAR(particle_vels_cpu[0][0], 1.0, 0.000001);
  EXPECT_NEAR(particle_vels_cpu[0][1], 1.0, 0.000001);

  EXPECT_NEAR(particle_pos_cpu[0][0], 0.2, 0.000001);
  EXPECT_NEAR(particle_pos_cpu[0][1], 0.3500, 0.000001);

#else

  EXPECT_NEAR(particle_vels_cpu[0][0], 1.0, 0.000001);
  EXPECT_NEAR(particle_pos_cpu[0][0], 0.2, 0.000001);

#endif

  musl_solver.P2G_double_mapping();

  cpu_array<Vectorr> node_moments_nt_cpu = musl_solver.nodes.moments_nt_gpu;

#if DIM == 3

  EXPECT_NEAR(node_moments_nt_cpu[0][0], 0.189, 0.000001);
  EXPECT_NEAR(node_moments_nt_cpu[0][1], 0.189, 0.000001);
  EXPECT_NEAR(node_moments_nt_cpu[0][2], 0.189, 0.000001);

  EXPECT_NEAR(node_moments_nt_cpu[1][0], 0.021, 0.000001);
  EXPECT_NEAR(node_moments_nt_cpu[1][1], 0.021, 0.000001);
  EXPECT_NEAR(node_moments_nt_cpu[1][2], 0.021, 0.000001);

#elif DIM == 2

  EXPECT_NEAR(node_moments_nt_cpu[0][0], 0.2700, 0.000001);
  EXPECT_NEAR(node_moments_nt_cpu[0][1], 0.27, 0.000001);
  EXPECT_NEAR(node_moments_nt_cpu[1][0], 0.03, 0.000001);
  EXPECT_NEAR(node_moments_nt_cpu[1][1], 0.03, 0.000001);
#else

  EXPECT_NEAR(node_moments_nt_cpu[0][0], 0.36000000000000004, 0.000001);
  EXPECT_NEAR(node_moments_nt_cpu[1][0], 0.03999999, 0.000001);

#endif

  musl_solver.G2P();

  cpu_array<Real> volumes_cpu = musl_solver.particles.volumes_gpu;

  cpu_array<Matrixr> F_cpu = musl_solver.particles.F_gpu;

#if DIM == 3
  EXPECT_NEAR(volumes_cpu[0], 0.69999999, 0.000001);
  EXPECT_NEAR(F_cpu[0](0), 1, 0.000001);
  EXPECT_NEAR(F_cpu[0](4), 1, 0.000001);
  EXPECT_NEAR(F_cpu[0](8), 1, 0.000001);
#elif DIM == 2
  EXPECT_NEAR(volumes_cpu[0], 0.69999999, 0.000001);
  EXPECT_NEAR(F_cpu[0](0), 1, 0.000001);
  EXPECT_NEAR(F_cpu[0](3), 1, 0.000001);
#else
  EXPECT_NEAR(volumes_cpu[0], 0.69999999, 0.000001);
  EXPECT_NEAR(F_cpu[0][0], 1, 0.000001);
#endif
}