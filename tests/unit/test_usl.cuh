#include "pyroclastmpm/common/global_settings.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/particles/particles.cuh"
#include "pyroclastmpm/solver/usl/usl.cuh"

// Features to be tested
// [x] USL::USL (implicitly through solver, inherits)
// [x] USL::solve (implicitly through all other tests)
// [x] USL::G2P
// [x] USL::P2G

using namespace pyroclastmpm;

/**
 * @brief Construct a new TEST object for ParticlesContainer G2P transfer
 *
 */
TEST(USL, Solve)
{
  cpu_array<Matrix3r> stresses = std::vector({Matrix3r::Ones()});
#if DIM == 3
  cpu_array<Vectorr> pos = std::vector({Vectorr({0.1, 0.25, 0.3}), Vectorr({0.1, 0.25, 0.3})

  });
#elif DIM == 2
  cpu_array<Vectorr> pos = std::vector({Vectorr({0.1, 0.25}),
                                        Vectorr({0.1, 0.25})});
#else
  cpu_array<Vectorr> pos = std::vector({Vectorr(0.1), Vectorr(0.1)});
#endif

  cpu_array<Vectorr> vels = std::vector({Vectorr::Ones(), Vectorr::Ones()});

  cpu_array<Real> volumes = std::vector({0.7, 0.4});

  cpu_array<Real> masses = std::vector({0.1, 0.3});

  set_globals(0.1, 1, LinearShapeFunction, "output");

  ParticlesContainer particles = ParticlesContainer(pos, vels);

  NodesContainer nodes = NodesContainer(Vectorr::Zero(), Vectorr::Ones(), 1.0);

  USL usl_solver = USL(particles, nodes);

  LinearElastic mat = LinearElastic(1000, 0.1, 0.1);
  usl_solver.materials.push_back(mat);

  usl_solver.particles.masses_gpu = masses;
  usl_solver.particles.volumes_gpu = volumes;
  usl_solver.particles.volumes_original_gpu = volumes;
  usl_solver.particles.stresses_gpu = stresses;
  usl_solver.particles.partition();

  usl_solver.calculate_shape_function();

  usl_solver.P2G();

  cpu_array<Real> masses_cpu = usl_solver.nodes.masses_gpu;
  cpu_array<Vectorr> moments_cpu = usl_solver.nodes.moments_gpu;

#if DIM == 3
  // TODO test other 6 nodes (only 2 tested)
  const Real expected_node_mass[2] = {0.189, 0.020999999};
  EXPECT_NEAR(masses_cpu[0], expected_node_mass[0], 0.000001);
  EXPECT_NEAR(masses_cpu[1], expected_node_mass[1], 0.000001);

  const Vectorr expected_node_moment[2] = {Vectorr({0.189, 0.189, 0.189}), Vectorr({0.02099999, 0.02099999, 0.02099999})};
  EXPECT_NEAR(moments_cpu[0][0], expected_node_moment[0][0], 0.000001);
  EXPECT_NEAR(moments_cpu[0][1], expected_node_moment[0][1], 0.000001);
  EXPECT_NEAR(moments_cpu[0][2], expected_node_moment[0][2], 0.000001);
  EXPECT_NEAR(moments_cpu[1][0], expected_node_moment[1][0], 0.000001);
  EXPECT_NEAR(moments_cpu[1][1], expected_node_moment[1][1], 0.000001);
  EXPECT_NEAR(moments_cpu[1][2], expected_node_moment[1][2], 0.000001);
#elif DIM == 2
  // TODO test other 2 nodes (only 2 tested)
  const Real expected_node_mass[2] = {0.27, 0.03};
  EXPECT_NEAR(masses_cpu[0], expected_node_mass[0], 0.000001);
  EXPECT_NEAR(masses_cpu[1], expected_node_mass[1], 0.000001);

  const Vectorr expected_node_moment[2] = {Vectorr({0.27, 0.27}), Vectorr({0.03, 0.03})};
  EXPECT_NEAR(moments_cpu[0][0], expected_node_moment[0][0], 0.000001);
  EXPECT_NEAR(moments_cpu[0][1], expected_node_moment[0][1], 0.000001);
  EXPECT_NEAR(moments_cpu[1][0], expected_node_moment[1][0], 0.000001);
  EXPECT_NEAR(moments_cpu[1][1], expected_node_moment[1][1], 0.000001);
#else
  const Real expected_node_mass[2] = {0.36, 0.04};
  EXPECT_NEAR(masses_cpu[0], expected_node_mass[0], 0.000001);
  EXPECT_NEAR(masses_cpu[1], expected_node_mass[1], 0.000001);

  const Vectorr expected_node_moment[2] = {Vectorr(0.36), Vectorr(0.04)};
  EXPECT_NEAR(moments_cpu[0][0], expected_node_moment[0][0], 0.000001);
  EXPECT_NEAR(moments_cpu[1][0], expected_node_moment[1][0], 0.000001);
#endif

  usl_solver.nodes.integrate();

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
  const Matrixr expected_F = Matrixr({{0.805555555, 0.0, 0.0}, {0.0, 0.90666666, 0.0}, {0.0, 0.0, 0.916666}});

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

  const Vectorr expected_position = Vectorr({0.2, 0.35});
  const Vectorr expected_velocities = Vectorr({1.0, 1.0});

  // TODO tests non diagonal F
  const Matrixr expected_F = Matrixr({{0.805555555, 0.0}, {0.0, 0.90666666}});

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