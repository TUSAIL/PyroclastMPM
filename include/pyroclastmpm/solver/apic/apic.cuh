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
#include "pyroclastmpm/solver/apic/apic_kernels.cuh"
#include "pyroclastmpm/solver/solver.cuh"
namespace pyroclastmpm {

/*! @brief Update Stress Last (USL) solver */
struct APIC : Solver {
 public:
  using Solver::Solver;

  // APIC(std::shared_ptr<ParticlesContainer> _particles_ptr = NULL,
  //               std::shared_ptr<NodesContainer> _nodes_ptr = NULL,
  //               thrust::host_vector<BoundaryCondition*> _boundaryconditions =
  //                   thrust::host_vector<BoundaryCondition*>());
  APIC(ParticlesContainer _particles,
       NodesContainer _nodes,
       thrust::host_vector<MaterialType> _materials =
           thrust::host_vector<MaterialType>(),
       thrust::host_vector<BoundaryConditionType> _boundaryconditions =
           thrust::host_vector<BoundaryConditionType>());
  // FUNCTIONS
  /*!
   * @brief Particle to grid update
   */
  void P2G();

  /*! @brief Grid to particle update */
  void G2P();

  /*!
   * @brief  Solver one iteration
   */
  void solve() override;

  /*!
   * @brief reset temporary arrays
   */
  void reset() override;

  // VARIABLES

  /*! @brief particles' velocity gradients */
  // thrust::device_vector<Matrix3r> Bp_gpu;

  Matrix3r Wp_inverse;
};

}  // namespace pyroclastmpm