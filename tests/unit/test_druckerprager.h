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

#pragma once

#include "pyroclastmpm/materials/druckerprager/druckerpragermat.h"

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

  EXPECT_NEAR(mat.lame_modulus, 277.777771, 0.0001);
}