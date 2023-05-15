#pragma once

#include "pyroclastmpm/materials/druckerprager/druckerpragermat.cuh"


// Functions to test
// [x] DruckerPrager::DruckerPrager
// [ ] DruckerPrager::stress_update (disconnected)
// [ ] DruckerPrager::stress_update (elastic)
// [ ] DruckerPrager::stress_update (plastic)



/**
 * @brief Construct a new test to test drucker prager constructor
 *
 */
TEST(DruckerPrager, TestConstructor) {
  DruckerPrager mat = DruckerPrager(1000, 1000, 0.2, 0.1, 0.1, 0.1);

  EXPECT_NEAR(mat.density, 1000., 0.001);
  EXPECT_NEAR(mat.E, 1000., 0.001);
  EXPECT_NEAR(mat.pois, 0.2, 0.001);
  EXPECT_NEAR(mat.friction_angle, 0.1, 0.001);
  EXPECT_NEAR(mat.cohesion, 0.1, 0.001);
  EXPECT_NEAR(mat.vcs, 0.1, 0.001);

  EXPECT_NEAR(mat.shear_modulus, 416.666656, 0.001);

  EXPECT_NEAR(mat.lame_modulus,  277.777771, 0.0001);

}
