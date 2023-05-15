#pragma once

#include "pyroclastmpm/boundaryconditions/bodyforce/bodyforce.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"

#include "pyroclastmpm/particles/particles.cuh"

// Functions to test
// [x] BodyForce::BodyForce, mode 0, forces (implicitly through apply_on_nodes_f_ext)
// [x] BodyForce::BodyForce, mode 1, moments (implicitly through apply_on_nodes_moments, mode1 )
// [x] BodyForce::BodyForce, mode 2, fixed (implicitly through apply_on_nodes_fixed, mode 2)
// [x] BodyForce::apply_on_nodes_f_ext, mode 0, forces
// [x] BodyForce::apply_on_nodes_moments, mode 1, moments
// [x] BodyForce::apply_on_nodes_moments, mode 2, fixed

using namespace pyroclastmpm;

/**
 * @brief Construct a new TEST object for the BodyForce boundary condition
 *
 */
TEST(BodyForce, ApplyOnNodesForces)
{
    // TEST x-axis

    std::vector<bool> mask = {false, true};

#if DIM == 3
    std::vector<Vectorr> values = {Vectorr({0., 0.25, 0.}), Vectorr({0.8, 0.6, 0.4})};
#elif DIM == 2
    std::vector<Vectorr> values = {Vectorr({0., 0.25}), Vectorr({0.8, 0.6})};
#else
    std::vector<Vectorr> values = {Vectorr(0.), Vectorr(0.8)};
#endif

    BodyForce boundarycondition = BodyForce("forces", values, mask);

    EXPECT_EQ(boundarycondition.mode_id, 0);

    Vectorr min = Vectorr::Zero();
    Vectorr max = Vectorr::Ones();

    Real nodal_spacing = 0.5;

    NodesContainer nodes = NodesContainer(min, max, nodal_spacing);

    // this might be needed if we add particles to the apply_on_nodes_f_ext argument later
    // ParticlesContainer particles = ParticlesContainer(std::vector({Vectorr::Ones() * 0.1}));

    //  this part just checks if forces are added (not fixed)
    nodes.forces_external_gpu[0] = values[0];
    nodes.forces_external_gpu[1] = values[1];

    boundarycondition.apply_on_nodes_f_ext(nodes);

    cpu_array<Vectorr> forces_ext = nodes.forces_external_gpu;

#if DIM == 3
    EXPECT_NEAR(forces_ext[0][0], 0., 0.0001);
    EXPECT_NEAR(forces_ext[0][1], 0.25, 0.0001);
    EXPECT_NEAR(forces_ext[0][2], 0., 0.0001);

    EXPECT_NEAR(forces_ext[1][0], 2. * values[1][0], 0.0001);
    EXPECT_NEAR(forces_ext[1][1], 2. * values[1][1], 0.0001);
    EXPECT_NEAR(forces_ext[1][2], 2. * values[1][2], 0.0001);
#elif DIM == 2
    EXPECT_NEAR(forces_ext[0][0], 0., 0.0001);
    EXPECT_NEAR(forces_ext[0][1], 0.25, 0.0001);
    EXPECT_NEAR(forces_ext[1][0], 2. * values[1][0], 0.0001);
    EXPECT_NEAR(forces_ext[1][1], 2. * values[1][1], 0.0001);

#else
    EXPECT_NEAR(forces_ext[0][0], 0., 0.0001);
    EXPECT_NEAR(forces_ext[1][0], 2. * values[1][0], 0.0001);
#endif
}

/**
 * @brief Construct a new TEST object for the BodyForce boundary condition
 *
 */
TEST(BodyForce, ApplyOnNodesMoments)
{
    // TEST x-axis

    std::vector<bool> mask = {false, true};

#if DIM == 3
    std::vector<Vectorr> values = {Vectorr({0., 0.25, 0.}), Vectorr({0.8, 0.6, 0.4})};
#elif DIM == 2
    std::vector<Vectorr> values = {Vectorr({0., 0.25}), Vectorr({0.8, 0.6})};
#else
    std::vector<Vectorr> values = {Vectorr(0.), Vectorr(0.8)};
#endif

    BodyForce boundarycondition = BodyForce("moments", values, mask);

    EXPECT_EQ(boundarycondition.mode_id, 1);

    Vectorr min = Vectorr::Zero();
    Vectorr max = Vectorr::Ones();

    Real nodal_spacing = 0.5;

    NodesContainer nodes = NodesContainer(min, max, nodal_spacing);

    ParticlesContainer particles = ParticlesContainer(std::vector({Vectorr::Ones() * 0.1}));

    //  this part just checks if moments are added (not fixed)
    nodes.moments_gpu[0] = values[0];
    nodes.moments_gpu[1] = values[1];

    nodes.moments_nt_gpu[0] = values[0];
    nodes.moments_nt_gpu[1] = values[1];

    boundarycondition.apply_on_nodes_moments(nodes, particles);

    cpu_array<Vectorr> moments = nodes.moments_gpu;
    cpu_array<Vectorr> moments_nt = nodes.moments_nt_gpu;

#if DIM == 3
    EXPECT_NEAR(moments[0][0], 0., 0.0001);
    EXPECT_NEAR(moments[0][1], 0.25, 0.0001);
    EXPECT_NEAR(moments[0][2], 0., 0.0001);

    EXPECT_NEAR(moments[1][0], 2. * values[1][0], 0.0001);
    EXPECT_NEAR(moments[1][1], 2. * values[1][1], 0.0001);
    EXPECT_NEAR(moments[1][2], 2. * values[1][2], 0.0001);
#elif DIM == 2
    EXPECT_NEAR(moments[0][0], 0., 0.0001);
    EXPECT_NEAR(moments[0][1], 0.25, 0.0001);
    EXPECT_NEAR(moments[1][0], 2. * values[1][0], 0.0001);
    EXPECT_NEAR(moments[1][1], 2. * values[1][1], 0.0001);

#else
    EXPECT_NEAR(moments[0][0], 0., 0.0001);
    EXPECT_NEAR(moments[1][0], 2. * values[1][0], 0.0001);
#endif
}

/**
 * @brief Construct a new TEST object for the BodyForce boundary condition
 *
 */
TEST(BodyForce, ApplyOnNodesMomentsFixed)
{
    // TEST x-axis

    std::vector<bool> mask = {false, true};

#if DIM == 3
    std::vector<Vectorr> values = {Vectorr({0., 0.25, 0.}), Vectorr({0.8, 0.6, 0.4})};
#elif DIM == 2
    std::vector<Vectorr> values = {Vectorr({0., 0.25}), Vectorr({0.8, 0.6})};
#else
    std::vector<Vectorr> values = {Vectorr(0.), Vectorr(0.8)};
#endif

    BodyForce boundarycondition = BodyForce("fixed", values, mask);

    EXPECT_EQ(boundarycondition.mode_id, 2);

    Vectorr min = Vectorr::Zero();
    Vectorr max = Vectorr::Ones();

    Real nodal_spacing = 0.5;

    NodesContainer nodes = NodesContainer(min, max, nodal_spacing);

    ParticlesContainer particles = ParticlesContainer(std::vector({Vectorr::Ones() * 0.1}));

    //  this part just checks if moments are added (not fixed)
    nodes.moments_gpu[0] = Vectorr::Zero();
    nodes.moments_gpu[1] = Vectorr::Zero();

    nodes.moments_nt_gpu[0] = Vectorr::Zero();
    nodes.moments_nt_gpu[1] = Vectorr::Zero();

    boundarycondition.apply_on_nodes_moments(nodes, particles);

    cpu_array<Vectorr> moments = nodes.moments_gpu;
    cpu_array<Vectorr> moments_nt = nodes.moments_nt_gpu;

#if DIM == 3
    EXPECT_NEAR(moments[0][0], 0., 0.0001);
    EXPECT_NEAR(moments[0][1], 0., 0.0001);
    EXPECT_NEAR(moments[0][2], 0., 0.0001);

    EXPECT_NEAR(moments[1][0], values[1][0], 0.0001);
    EXPECT_NEAR(moments[1][1], values[1][1], 0.0001);
    EXPECT_NEAR(moments[1][2], values[1][2], 0.0001);

    EXPECT_NEAR(moments_nt[0][0], 0., 0.0001);
    EXPECT_NEAR(moments_nt[0][1], 0., 0.0001);
    EXPECT_NEAR(moments_nt[0][2], 0., 0.0001);

    EXPECT_NEAR(moments_nt[1][0], values[1][0], 0.0001);
    EXPECT_NEAR(moments_nt[1][1], values[1][1], 0.0001);
    EXPECT_NEAR(moments_nt[1][2], values[1][2], 0.0001);
#elif DIM == 2
    EXPECT_NEAR(moments[0][0], 0., 0.0001);
    EXPECT_NEAR(moments[0][1], 0., 0.0001);
    EXPECT_NEAR(moments[1][0], values[1][0], 0.0001);
    EXPECT_NEAR(moments[1][1], values[1][1], 0.0001);

    EXPECT_NEAR(moments_nt[0][0], 0., 0.0001);
    EXPECT_NEAR(moments_nt[0][1], 0., 0.0001);
    EXPECT_NEAR(moments_nt[1][0], values[1][0], 0.0001);
    EXPECT_NEAR(moments_nt[1][1], values[1][1], 0.0001);

#else
    EXPECT_NEAR(moments[0][0], 0., 0.0001);
    EXPECT_NEAR(moments[1][0], values[1][0], 0.0001);

    EXPECT_NEAR(moments_nt[0][0], 0., 0.0001);
    EXPECT_NEAR(moments_nt[1][0], values[1][0], 0.0001);
#endif
}
