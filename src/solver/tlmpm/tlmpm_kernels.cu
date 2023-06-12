// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
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

#include "pyroclastmpm/solver/tlmpm/tlmpm_kernels.cuh"

namespace pyroclastmpm
{

    extern __constant__ Real dt_gpu;
    extern __constant__ int num_surround_nodes_gpu;
    extern __constant__ int forward_window_gpu[64][3];

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
                                     const int num_particles)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid >= num_particles)
        {
            return;
        } // block access threads

        const Vectori particle_bin = particles_bins_gpu[tid];

        Vectorr vel_inc = Vectorr::Zero();
        Vectorr dvel_inc = Vectorr::Zero();
        Matrixr F_dot = Matrixr::Zero();

#pragma unroll
        for (int i = 0; i < num_surround_nodes_gpu; i++)
        {
#if DIM == 3
            const Vectori forward_mesh = Vectori({forward_window_gpu[i][0],
                                                  forward_window_gpu[i][1],
                                                  forward_window_gpu[i][2]});

            const Vectori selected_bin = particle_bin + forward_mesh;

            const unsigned int nhash = selected_bin[0] +
                                       selected_bin[1] * num_cells[0] +
                                       selected_bin[2] * num_cells[0] * num_cells[1];

#elif DIM == 2
            const Vectori forward_mesh = Vectori({forward_window_gpu[i][0],
                                                  forward_window_gpu[i][1]});

            const Vectori selected_bin = particle_bin + forward_mesh;

            const unsigned int nhash = selected_bin[0] +
                                       selected_bin[1] * num_cells[0];
#else
            const Vectori forward_mesh = Vectori(forward_window_gpu[i][0]);

            const Vectori selected_bin = particle_bin + forward_mesh;

            const unsigned int nhash = selected_bin[0];
#endif
            bool invalidCell = false;

#pragma unroll
            for (int axis = 0; axis < DIM; axis++)
            {
                if ((selected_bin[axis] < 0) || (selected_bin[axis] >= num_cells[axis]))
                {
                    invalidCell = true;
                    break;
                }
            }

            if (invalidCell)
            {
                continue;
            }

            const Real node_mass = nodes_masses_gpu[nhash];
            if (node_mass <= 0.000000001)
            {
                continue;
            }

            const Vectorr dpsi_particle =
                particles_dpsi_gpu[tid * num_surround_nodes_gpu + i];

            const Vectorr node_velocity_nt = nodes_moments_nt_gpu[nhash] / node_mass;

            const Real psi_particle =
                particles_psi_gpu[tid * num_surround_nodes_gpu + i];

            vel_inc += psi_particle * node_velocity_nt;
            F_dot +=
                dpsi_particle *
                node_velocity_nt.transpose();
        }

        particles_positions_gpu[tid] += dt_gpu * vel_inc;
        particles_F_gpu[tid] += dt_gpu * F_dot;

        const Matrixr F_inv = particles_F_gpu[tid].inverse();

        particles_velocity_gradients_gpu[tid] = F_dot * F_inv;

        Real J = particles_F_gpu[tid].determinant();

        particles_volumes_gpu[tid] = J * particles_volumes_original_gpu[tid];
    }

    __global__ void KERNEL_TLMPM_CONVERT_STRESS(
        Matrix3r *particles_stresses_pk1_gpu,
        const Matrix3r *particles_stresses_gpu,
        const Matrixr *particles_F_gpu,
        const int num_particles)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid >= num_particles)
        {
            return;
        } // block access threads

#if DIM == 3
        Matrixr stress = particles_stresses_gpu[tid];
#else
        Matrix3r stress_3d = particles_stresses_gpu[tid];
        Matrixr stress = stress_3d.block(0, 0, DIM, DIM);
#endif

        const Matrixr F = particles_F_gpu[tid];
        const Matrixr F_inv = F.inverse();
        const Matrixr F_inv_T = F_inv.transpose();
        const Real J = F.determinant();

        stress = J * stress * F_inv_T;

#if DIM == 3
        particles_stresses_pk1_gpu[tid] = stress;
#else
        stress_3d.block(0, 0, DIM, DIM) = stress;
        particles_stresses_pk1_gpu[tid] = stress_3d;
#endif
    }

} // namespace pyroclastmpm