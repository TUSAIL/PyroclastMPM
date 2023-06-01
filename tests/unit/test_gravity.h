#pragma once

#include "pyroclastmpm/boundaryconditions/gravity.h"
#include "pyroclastmpm/nodes/nodes.h"

// Functions to test
// [x] Gravity::Gravity, no ramp (implicitly tested in apply_on_nodes_f_ext, no
// ramp) [x] Gravity::Gravity, with ramp (implicitly tested in
// apply_on_nodes_f_ext, with ramp) [x] Gravity::apply_on_nodes_f_ext, no ramp
// [ ] Gravity::apply_on_nodes_f_ext, with ramp

using namespace pyroclastmpm;

/**
 * @brief Construct a new TEST object for testing the gravity boundary condition
 *
 */
TEST(Gravity, APPLY_GRAVITY_NO_RAMP) {

  Vectorr gravity = Vectorr::Ones();
  gravity *= -9.81;

  Gravity boundarycond = Gravity(gravity);

  EXPECT_EQ(boundarycond.is_ramp, false);

  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real nodal_spacing = 0.5;

  NodesContainer nodes = NodesContainer(min, max, nodal_spacing);

  // this might be needed if we add particles to the apply_on_nodes_f_ext
  // argument later ParticlesContainer particles =
  // ParticlesContainer(std::vector({Vectorr::Ones()*0.1}));

  nodes.masses_gpu[0] = 1.;
  nodes.masses_gpu[1] = 2.;

  boundarycond.apply_on_nodes_f_ext(nodes);

  cpu_array<Vectorr> forces_external_cpu = nodes.forces_external_gpu;

  EXPECT_NEAR(forces_external_cpu[0][0], -9.81, 0.000001);
  EXPECT_NEAR(forces_external_cpu[1][0], -19.62, 0.000001);

#if DIM > 1
  EXPECT_NEAR(forces_external_cpu[0][1], -9.81, 0.000001);
  EXPECT_NEAR(forces_external_cpu[1][1], -19.62, 0.000001);
#endif

#if DIM > 2
  EXPECT_NEAR(forces_external_cpu[0][2], -9.81, 0.000001);
  EXPECT_NEAR(forces_external_cpu[1][2], -19.62, 0.000001);
#endif
}

TEST(Gravity, APPLY_GRAVITY_WITH_RAMP) {

  Vectorr gravity = Vectorr::Ones();
  gravity *= -9.81;

  // give 2 steps to ramp up to full gravity (gravity start at 0. m.s^-2)
  Gravity boundarycond = Gravity(Vectorr::Zero(), true, 2, gravity);

  EXPECT_EQ(boundarycond.is_ramp, true);

  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real nodal_spacing = 0.5;

  NodesContainer nodes = NodesContainer(min, max, nodal_spacing);

  // this might be needed if we add particles to the apply_on_nodes_f_ext
  // argument later (for NGF?) ParticlesContainer particles =
  // ParticlesContainer(std::vector({Vectorr::Ones()*0.1}));

  nodes.masses_gpu[0] = 1.;
  nodes.masses_gpu[1] = 2.;

  set_global_step(1);
  boundarycond.apply_on_nodes_f_ext(nodes);

  cpu_array<Vectorr> forces_external_cpu = nodes.forces_external_gpu;

  EXPECT_NEAR(forces_external_cpu[0][0], -4.905, 0.000001);
  EXPECT_NEAR(forces_external_cpu[1][0], -9.81, 0.000001);

  set_global_step(2);
  boundarycond.apply_on_nodes_f_ext(nodes);

  forces_external_cpu = nodes.forces_external_gpu;

  EXPECT_NEAR(forces_external_cpu[0][0], -9.81, 0.000001);
  EXPECT_NEAR(forces_external_cpu[1][0], -19.62, 0.000001);
}