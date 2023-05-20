#include <gtest/gtest.h>

#include "pyroclastmpm/common/global_settings.cuh"

// Global Settings
#include "unit/test_global.cuh" // Tests needed

// Partitioning
#include "unit/test_spatialpartition.cuh"

// // Materials
#include "unit/test_linearelastic.cuh"
#include "unit/test_newtonfluid.cuh" // Tests needed 
#include "unit/test_localrheo.cuh" // Tests needed

// // Boundary conditions
#include "unit/test_gravity.cuh"
// #include "unit/test_bodyforce.cuh"
// #include "unit/test_rigidparticles.cuh"
// #include "unit/test_planardomain.cuh"

// Particles, Nodes, Shape Functions
#include "unit/test_particles.cuh"
#include "unit/test_nodes.cuh"
#include "unit/test_shapefunctions.cuh"

// // Solvers
#include "unit/test_solver.cuh"
#include "unit/test_usl.cuh"
// #include "unit/test_musl.cuh"
// #include "unit/test_tlmpm.cuh"

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}