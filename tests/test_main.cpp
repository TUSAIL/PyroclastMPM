#include <gtest/gtest.h>

#include "pyroclastmpm/common/global_settings.h"

// Global Settings
#include "unit/test_global.h" // Tests needed

// Partitioning
#include "unit/test_spatialpartition.h"

// Materials
#include "unit/test_linearelastic.h"
#include "unit/test_localrheo.h"   // Tests needed
#include "unit/test_newtonfluid.h" // Tests needed

// Boundary conditions
#include "unit/test_bodyforce.h"
#include "unit/test_gravity.h"
#include "unit/test_nodedomain.h"
#include "unit/test_planardomain.h"
#include "unit/test_rigidbodylevelset.h"

// // Particles, Nodes, Shape Functions
#include "unit/test_nodes.h"
#include "unit/test_particles.h"
#include "unit/test_shapefunctions.h"

// // // Solvers
#include "unit/test_solver.h"
#include "unit/test_usl.h"
// #include "unit/test_musl.cuh"
// #include "unit/test_tlmpm.cuh"

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}