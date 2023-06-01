#ifndef TEST_NONLOCALNGF_H
#define TEST_NONLOCALNGF_H

#include "pyroclastmpm/materials/nonlocalngf/nonlocalngfmat.h"

namespace pyroclastmpm {

/**
 * @brief Construct a new Non Local NGF:: Non Local NGF object
 *
 */
TEST(NonLocalNGF, Constructor) {
  NonLocalNGF mat(1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  EXPECT_EQ(mat.name, "NonLocalNGF");
}

} // namespace pyroclastmpm
#endif