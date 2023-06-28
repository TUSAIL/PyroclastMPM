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

#include "pyroclastmpm/materials/newtonfluid.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern const Real __constant__ dt_gpu;
#else
extern const Real dt_cpu;
#endif

///@brief Stress update for Newtonian fluid
///@param particles_stresses_gpu stress tensor
///@param particles_velocity_gradients_gpu velocity gradient
///@param particles_masses_gpu particle masses
///@param particles_volumes_gpu particle volumes (updated)
///@param particles_volumes_original_gpu particle volumes (original)
///@param particles_colors_gpu particle colors (or material ids)
///@param particles_is_active_gpu flag if has active status
///@param viscosity fluid viscosity
///@param bulk_modulus bulk modulus
///@param gamma gamma (7 for water and 1.4 for air)
///@param mat_id material id
///@param tid thread id
__device__ __host__ inline void stress_update_newtonfluid(
    Matrix3r *particles_stresses_gpu,
    const Matrixr *particles_velocity_gradients_gpu,
    const Real *particles_masses_gpu, const Real *particles_volumes_gpu,
    const Real *particles_volumes_original_gpu,
    const uint8_t *particles_colors_gpu, const bool *particles_is_active_gpu,
    const Real bulk_viscosity, const Real bulk_modulus, const Real gamma,
    const int mat_id, const int tid) {

  if (!particles_is_active_gpu[tid]) {
    return;
  }

  if (particles_colors_gpu[tid] != mat_id) {
    return;
  }

  // isotropic stress tensor

  const Real density = particles_masses_gpu[tid] / particles_volumes_gpu[tid];

  const Real density_original =
      particles_masses_gpu[tid] / particles_volumes_original_gpu[tid];

  const Real pressure =
      -bulk_modulus *
      ((Real)pow(density / density_original, gamma) - (Real)1.0);

  Matrix3r isotropic_part = Matrix3r::Zero();

  isotropic_part.block(0, 0, DIM, DIM) = pressure * Matrixr::Identity();
  // deviatoric stress tensor

  const Matrixr vel_grad = particles_velocity_gradients_gpu[tid];

  const Matrixr strain_rate = 0.5 * (vel_grad + vel_grad.transpose());

  const Real volumetric_strain_rate = strain_rate.trace();

  Matrix3r deviatoric_part = Matrix3r::Zero();

  deviatoric_part.block(0, 0, DIM, DIM) =
      strain_rate - (1. / 3.) * volumetric_strain_rate * Matrixr::Identity();

  deviatoric_part *= (Real)(2. * bulk_viscosity / 3.);

  particles_stresses_gpu[tid] = isotropic_part + deviatoric_part;
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_STRESS_UPDATE_NEWTONFLUID(
    Matrix3r *particles_stresses_gpu,
    const Matrixr *particles_velocity_gradients_gpu,
    const Real *particles_masses_gpu, const Real *particles_volumes_gpu,
    const Real *particles_volumes_original_gpu,
    const uint8_t *particles_colors_gpu, const bool *particles_is_active_gpu,
    const int num_particles, const Real viscosity, const Real bulk_modulus,
    const Real gamma, const int mat_id) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  stress_update_newtonfluid(
      particles_stresses_gpu, particles_velocity_gradients_gpu,
      particles_masses_gpu, particles_volumes_gpu,
      particles_volumes_original_gpu, particles_colors_gpu,
      particles_is_active_gpu, viscosity, bulk_modulus, gamma, mat_id, tid);
}

#endif

} // namespace pyroclastmpm