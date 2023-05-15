#include "pyroclastmpm/common/global_settings.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/particles/particles.cuh"
#include "pyroclastmpm/solver/tlmpm/tlmpm.cuh"

// Features to be tested
// [x] TLMPM::TLMPM (implicitly through solver, inherits)
// [x] TLMPM::solve (implicitly through all other tests)
// [x] TLMPM::G2P
// [x] TLMPM::CauchyStressToPK1Stress
using namespace pyroclastmpm;

/**
 * @brief Construct a new TEST object for ParticlesContainer G2P transfer
 *
 */
TEST(TLMPM, Solve)

{

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

    TLMPM tlmpm_solver = TLMPM(particles, nodes);

    LinearElastic mat = LinearElastic(1000, 0.1, 0.1);
    tlmpm_solver.materials.push_back(mat);

    tlmpm_solver.P2G();

    tlmpm_solver.nodes.integrate();

    tlmpm_solver.G2P_double_mapping();

    tlmpm_solver.P2G_double_mapping();

    tlmpm_solver.G2P();

    cpu_array<Real> volumes_cpu = tlmpm_solver.particles.volumes_gpu;

    cpu_array<Matrixr> F_cpu = tlmpm_solver.particles.F_gpu;

#if DIM == 3
    EXPECT_NEAR(volumes_cpu[0], 1., 0.000001);
    EXPECT_NEAR(F_cpu[0](0), 1, 0.000001);
    EXPECT_NEAR(F_cpu[0](4), 1, 0.000001);
    EXPECT_NEAR(F_cpu[0](8), 1, 0.000001);
#elif DIM == 2
    EXPECT_NEAR(volumes_cpu[0], 1., 0.000001);
    EXPECT_NEAR(F_cpu[0](0), 1, 0.000001);
    EXPECT_NEAR(F_cpu[0](3), 1, 0.000001);
#else
    EXPECT_NEAR(volumes_cpu[0], 1, 0.000001);
    EXPECT_NEAR(F_cpu[0][0], 1, 0.000001);
#endif

    // prescribe stress
    cpu_array<Matrix3r> stresses = std::vector({
        Matrix3r({{4., 2.0, 3.}, {0.1, 1, 5}, {6, 7, 8}}),
        Matrix3r({{4., 2.0, 3.}, {0.1, 1, 5}, {6, 7, 8}}),
    });

    tlmpm_solver.particles.stresses_gpu = stresses;
    tlmpm_solver.CauchyStressToPK1Stress();

    cpu_array<Matrix3r> stresses_cpu = tlmpm_solver.stresses_pk1_gpu;

#if DIM == 3
    EXPECT_NEAR(stresses_cpu[0](0), 4, 0.000001);
    EXPECT_NEAR(stresses_cpu[0](4), 1, 0.000001);
    EXPECT_NEAR(stresses_cpu[0](8), 8, 0.000001);
#elif DIM == 2
    EXPECT_NEAR(stresses_cpu[0](0), 4, 0.000001);
    EXPECT_NEAR(stresses_cpu[0](3), 2, 0.000001);
#else
    EXPECT_NEAR(stresses_cpu[0](0), 4, 0.000001);
#endif
}