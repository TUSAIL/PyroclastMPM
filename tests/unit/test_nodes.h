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

#include "pyroclastmpm/nodes/nodes.h"

/**
 * @brief Construct a new TEST object for NodesContainer to test the constructor
 *
 */
TEST(NodesContainer, CONSTRUCTOR)
{

  auto min = Vectorr::Zero();
  auto max = Vectorr::Ones();
  Real nodal_spacing = 0.5;

  auto nodes = pyroclastmpm::NodesContainer(min, max, nodal_spacing);

  cpu_array<Vectori> node_ids = nodes.node_ids_gpu;

#if DIM == 3
  EXPECT_EQ(nodes.grid.num_cells_total, 27);
  EXPECT_EQ(nodes.grid.num_cells[0], 3);
  EXPECT_EQ(nodes.grid.num_cells[1], 3);
  EXPECT_EQ(nodes.grid.num_cells[2], 3);
  EXPECT_EQ(node_ids[0], Vectori({0, 0, 0}));
  EXPECT_EQ(node_ids[1], Vectori({1, 0, 0}));
  EXPECT_EQ(node_ids[2], Vectori({2, 0, 0}));
  EXPECT_EQ(node_ids[3], Vectori({0, 1, 0}));
  EXPECT_EQ(node_ids[4], Vectori({1, 1, 0}));
  EXPECT_EQ(node_ids[5], Vectori({2, 1, 0}));
  EXPECT_EQ(node_ids[6], Vectori({0, 2, 0}));
  EXPECT_EQ(node_ids[7], Vectori({1, 2, 0}));
  EXPECT_EQ(node_ids[8], Vectori({2, 2, 0}));

  EXPECT_EQ(node_ids[9], Vectori({0, 0, 1}));
  EXPECT_EQ(node_ids[10], Vectori({1, 0, 1}));
  EXPECT_EQ(node_ids[11], Vectori({2, 0, 1}));
  EXPECT_EQ(node_ids[12], Vectori({0, 1, 1}));
  EXPECT_EQ(node_ids[13], Vectori({1, 1, 1}));
  EXPECT_EQ(node_ids[14], Vectori({2, 1, 1}));
  EXPECT_EQ(node_ids[15], Vectori({0, 2, 1}));
  EXPECT_EQ(node_ids[16], Vectori({1, 2, 1}));
  EXPECT_EQ(node_ids[17], Vectori({2, 2, 1}));

  EXPECT_EQ(node_ids[18], Vectori({0, 0, 2}));
  EXPECT_EQ(node_ids[19], Vectori({1, 0, 2}));
  EXPECT_EQ(node_ids[20], Vectori({2, 0, 2}));
  EXPECT_EQ(node_ids[21], Vectori({0, 1, 2}));
  EXPECT_EQ(node_ids[22], Vectori({1, 1, 2}));
  EXPECT_EQ(node_ids[23], Vectori({2, 1, 2}));
  EXPECT_EQ(node_ids[24], Vectori({0, 2, 2}));
  EXPECT_EQ(node_ids[25], Vectori({1, 2, 2}));
  EXPECT_EQ(node_ids[26], Vectori({2, 2, 2}));

#elif DIM == 2

  EXPECT_EQ(nodes.grid.num_cells_total, 9);
  EXPECT_EQ(nodes.grid.num_cells[0], 3);
  EXPECT_EQ(nodes.grid.num_cells[1], 3);
  EXPECT_EQ(node_ids[0], Vectori({0, 0}));
  EXPECT_EQ(node_ids[1], Vectori({1, 0}));
  EXPECT_EQ(node_ids[2], Vectori({2, 0}));
  EXPECT_EQ(node_ids[3], Vectori({0, 1}));
  EXPECT_EQ(node_ids[4], Vectori({1, 1}));
  EXPECT_EQ(node_ids[5], Vectori({2, 1}));
  EXPECT_EQ(node_ids[6], Vectori({0, 2}));
  EXPECT_EQ(node_ids[7], Vectori({1, 2}));
  EXPECT_EQ(node_ids[8], Vectori({2, 2}));

#else // DIM == 1
  EXPECT_EQ(nodes.grid.num_cells_total, 3);
  EXPECT_EQ(nodes.grid.num_cells[0], 3);
  EXPECT_EQ(node_ids[0], Vectori(0));
  EXPECT_EQ(node_ids[1], Vectori(1));
  EXPECT_EQ(node_ids[2], Vectori(2));
#endif
}
