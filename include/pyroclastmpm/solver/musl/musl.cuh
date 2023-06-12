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

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/solver/usl/usl.cuh"
#include "pyroclastmpm/solver/usl/usl_kernels.cuh"
#include "pyroclastmpm/solver/musl/musl_kernels.cuh"

namespace pyroclastmpm
{

  /*!
   * @brief Modified Update Stress Last Solver (MUSL)
   */
  struct MUSL : USL
  {
  public:

    explicit MUSL(
        ParticlesContainer _particles,
        NodesContainer _nodes,
        cpu_array<MaterialType> _materials =
            cpu_array<MaterialType>(),
        cpu_array<BoundaryConditionType> _boundaryconditions =
            cpu_array<BoundaryConditionType>(),
        Real _alpha = 0.99);

    /*!
     * @brief Particle to grid transfer 2
     */
    void P2G_double_mapping();

    /*!
     * @brief Grid to particle transfer 1
     */
    void G2P_double_mapping();

    /*!
     * @brief Grid to particles transfter 2
     */
    void G2P();

    /*!
     * @brief Solver one iteration
     */
    void solve() override;

  };

} // namespace pyroclastmpm