#pragma once

#include "pyroclastmpm/materials/linearelastic/linearelasticmat.cuh"

// Functions to test
// [x] LinearElastic::LinearElastic (Tested implicitly via stress_update)
// [x] LinearElastic::stress_update
// [ ] LinearElastic::calculate_timestep (TODO confirm correct formula and consistent with othe code)

TEST(LinearElastic, StressUpdateLinearElastic)
{

#if DIM == 3
  const std::vector<Vectorr> pos = {
      Vectorr({0.1, 0.1, 0.0})};
#elif DIM == 2
  const std::vector<Vectorr> pos = {
      Vectorr({0.1, 0.1})};
#else
  const std::vector<Vectorr> pos = {Vectorr(0.1)};
#endif
  const std::vector<Matrixr> velgrad = {Matrixr::Identity() * 0.1};
  set_global_dt(0.1);
  ParticlesContainer particles = ParticlesContainer(pos);

  LinearElastic mat = LinearElastic(1000, 0.1, 0.1);

  particles.velocity_gradient_gpu = velgrad;

  mat.stress_update(particles, 0);

  cpu_array<Matrix3r> stresses = particles.stresses_gpu;

#if DIM == 3
  Matrix3r expected_stress;
  expected_stress << 0.001023, 0, 0, 0,  0.001023, 0, 0, 0, 0.001023;
  EXPECT_NEAR(stresses[0](0), expected_stress(0), 0.0001);
  EXPECT_NEAR(stresses[0](4), expected_stress(4), 0.0001);
  EXPECT_NEAR(stresses[0](8), expected_stress(8), 0.0001);
#elif DIM == 2
  Matrix3r expected_stress;
  expected_stress << 0.001023, 0, 0, 0,  0.001023, 0, 0, 0, 0;
  EXPECT_NEAR(stresses[0](0), expected_stress(0), 0.0001);
  EXPECT_NEAR(stresses[0](4), expected_stress(4), 0.0001);
#else
  Matrix3r expected_stress;
  expected_stress << 0.001023, 0, 0, 0, 0, 0, 0, 0, 0;
  EXPECT_NEAR(stresses[0](0), expected_stress(0), 0.0001);
#endif
}