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

namespace pyroclastmpm {

__global__ void KERNELS_USL_P2G_APIC(
    Vector3r* nodes_moments_gpu,
    Vector3r* nodes_forces_internal_gpu,
    Real* nodes_masses_gpu,
    const Vector3i* node_ids_3d_gpu,
    const Matrix3r* particles_stresses_gpu,
    const Matrix3r* particles_velocity_gradients_gpu,
    const Vector3r* particles_velocities_gpu,
    const Vector3r* particles_positions_gpu,
    const Vector3r* particles_dpsi_gpu,
    const Real* particles_psi_gpu,
    const Real* particles_masses_gpu,
    const Real* particles_volumes_gpu,
    const int* particles_cells_start_gpu,
    const int* particles_cells_end_gpu,
    const int* particles_sorted_indices_gpu,
    const Vector3i num_nodes,
    const Real inv_cell_size,
    const int num_nodes_total);

__global__ void KERNEL_USL_G2P_APIC(Matrix3r* particles_velocity_gradients_gpu,
                               Matrix3r* particles_F_gpu,
                               Matrix3r* particles_strains_gpu,
                               Matrix3r* particles_strain_increments,
                               Vector3r* particles_velocities_gpu,
                               Vector3r* particles_positions_gpu,
                               Real* particles_volumes_gpu,
                               Real* particles_densities_gpu,
                               const Vector3r* particles_dpsi_gpu,
                               const Vector3i* particles_bins_gpu,
                               const Real* particles_volumes_original_gpu,
                               const Real* particles_densities_original_gpu,
                               const Real* particles_psi_gpu,
                               const Real* particles_masses_gpu,
                               const Vector3r* nodes_moments_gpu,
                               const Vector3r* nodes_moments_nt_gpu,
                               const Real* nodes_masses_gpu,
                               const Matrix3r Wp_inverse,
                               const Real inv_cell_size,
                               const Vector3i num_cells,
                               const int num_particles);

}  // namespace pyroclastmpm