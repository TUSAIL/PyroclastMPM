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

namespace pyroclastmpm
{

  // __global__ void KERNELS_TLMPM_P2G(Vector3r* nodes_moments_gpu,
  //                                 Vector3r* nodes_forces_internal_gpu,
  //                                 Real* nodes_masses_gpu,
  //                                 const Vector3i* node_ids_3d_gpu,
  //                                 const Matrix3r* particles_stresses_gpu,
  //                                 const Vector3r* particles_velocities_gpu,
  //                                 const Vector3r* particles_dpsi_gpu,
  //                                 const Real* particles_psi_gpu,
  //                                 const Real* particles_masses_gpu,
  //                                 const Real* particles_volumes_gpu,
  //                                 const int* particles_cells_start_gpu,
  //                                 const int* particles_cells_end_gpu,
  //                                 const int* particles_sorted_indices_gpu,
  //                                 const Vector3i num_nodes,
  //                                 const Real inv_cell_size,
  //                                 const int num_nodes_total);

  // __global__ void KERNEL_USL_TLMPM(Matrixr *particles_velocity_gradients_gpu,
  //                                Matrixr *particles_F_gpu,
  //                                Vectorr *particles_velocities_gpu,
  //                                Vectorr *particles_positions_gpu,
  //                                Real *particles_volumes_gpu,
  //                                const Vectorr *particles_dpsi_gpu,
  //                                const Vectori *particles_bins_gpu,
  //                                const Real *particles_volumes_original_gpu,
  //                                const Real *particles_psi_gpu,
  //                                const Real *particles_masses_gpu,
  //                                const Vectorr *nodes_moments_gpu,
  //                                const Vectorr *nodes_moments_nt_gpu,
  //                                const Real *nodes_masses_gpu,
  //                                const Vectori num_cells,
  //                                const int num_particles);

    __global__ void KERNEL_TLMPM_G2P(Matrixr *particles_velocity_gradients_gpu,
                                     Matrixr *particles_F_gpu,
                                     Real *particles_volumes_gpu,
                                     Vectorr *particles_positions_gpu,
                                     const Vectorr *nodes_moments_nt_gpu,
                                     const Vectorr *particles_dpsi_gpu,
                                     const Vectori *particles_bins_gpu,
                                     const Real *particles_volumes_original_gpu,
                                     const Real *particles_psi_gpu,
                                     const Real *particles_masses_gpu,
                                     const Real *nodes_masses_gpu,
                                     const Vectori num_cells,
                                     const int num_particles);

  __global__ void KERNEL_TLMPM_CONVERT_STRESS(
      Matrix3r *particles_stresses_pk1_gpu,
      const Matrix3r *particles_stresses_gpu,
      const Matrixr *particles_F_gpu,
      const int num_particles);

} // namespace pyroclastmpm