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

#include "pyroclastmpm/materials/muijop.h"

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
__device__ __host__ inline void
stress_update_muijop(Matrix3r *particles_stresses_gpu,
                     const Matrixr *particles_velocity_gradients_gpu,
                     const Real *particles_masses_gpu, Real *I_gpu,
                     const Real *particles_volumes_gpu,
                     const Real *particles_volumes_original_gpu,
                     const uint8_t *particles_colors_gpu,
                     const bool *particles_is_active_gpu,
                     const Real bulk_viscosity, const Real bulk_modulus,
                     const Real gamma, const int mat_id, const int tid) {

  // params temporary disabled
  // https://link.springer.com/article/10.1007/s10035-019-0886-6
  // (implementation of the model in the paper)

  // for hint of parameters see
  // https://github.com/neocpp89/mpm-2d/blob/10846575f09885b314e18f7eced9a925aba76e6f/materialsrc/g_local_mu2.c#L35
  const Real tol = 1e-12;
  // const Real d = 0.005; /// 0.53 mm ?
  const Real ps = 2450;
  // const Real mu_s = 0.38186286741;
  // const Real mu_2 = 0.64346838485;
  // const Real I0 = 0.279;

  // sand
  // const Real d = 0.005; // default
  const Real d = 0.5;
  const Real I0 = 0.278; // default
  // const Real I0 = 6.0;
  // const Real mu_s = 0.46188021535170065; // M=0.8
  const Real mu_s = 0.46188021535170065;
  // const Real mu_2 = 1.24820403635;
  const Real mu_2 = 0.6435;

  const Real alpha_r = 0.01;

  const Real alpha_s = 0.000001;

  const bool is_velgrad_strain_increment = false;

  if (!particles_is_active_gpu[tid]) {
    return;
  }

  if (particles_colors_gpu[tid] != mat_id) {
    return;
  }

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

  const Matrixr vel_grad = particles_velocity_gradients_gpu[tid];
  Matrixr strain_rate;
  if (is_velgrad_strain_increment) {
    // printf("dt: %.16f\n", dt);
    strain_rate = vel_grad / dt;
  } else {
    strain_rate = 0.5 * (vel_grad + vel_grad.transpose());
  }

  // isotropic stress tensor

  const Real density = particles_masses_gpu[tid] / particles_volumes_gpu[tid];

  const Real density_original =
      particles_masses_gpu[tid] / particles_volumes_original_gpu[tid];

  const Real pressure =
      std::max(bulk_modulus * (density / density_original - 1), tol);

  // const Real pressure =
  //     bulk_modulus *
  //     (particles_volumes_original_gpu[tid] - particles_volumes_gpu[tid]) /
  //     particles_volumes_original_gpu[tid];

  // deviatoric strain rate tensor
  const Matrixr dot_gamma_tensor =
      strain_rate - (strain_rate.trace() / 3.) * Matrixr::Identity();

  // second invariant of deviatoric strain rate tensor
  const Real dot_gamma = (Real)sqrt(
      0.5 * (dot_gamma_tensor * dot_gamma_tensor.transpose()).trace());

  Matrix3r tau = Matrix3r::Zero();

  // if (pressure > tol) {

  const Real I = (dot_gamma * d) / sqrt(pressure / ps);
  I_gpu[tid] = I;

  // printf("I: %f\n", I);
  // const Real mu_I = mu_s + (mu_2 - mu_s) / (1.0 + (I / I0));
  // printf("mu_I: %f\n", mu_I);
  // const Real nu = (mu_I * pressure) / dot_gamma;

  Real nu = mu_s * pressure * (1 - exp(-dot_gamma / alpha_r)) / dot_gamma;

  nu += (mu_2 - mu_s) / (1.0 + (I / I0)) * pressure * dot_gamma /
        sqrt(dot_gamma * dot_gamma + alpha_s * alpha_s);

  // printf("dot_gamma: %.16f\n", dot_gamma);
  // printf("nu: %.16f\n", nu);
  // printf("mu_I: %.16f\n", mu_I);

  Matrix3r dot_gamma_tensor_3d = Matrix3r::Zero();
  dot_gamma_tensor_3d.block(0, 0, DIM, DIM) = dot_gamma_tensor;
  tau = nu * dot_gamma_tensor_3d;
  // } else {
  //   const Real p = (1 / 3.) * particles_stresses_gpu[tid].trace();

  //   tau = particles_stresses_gpu[tid] - p * Matrix3r::Identity();
  // }
  // exit(0);
  // Matrix3r isotropic_part = Matrix3r::Zero();

  // isotropic_part.block(0, 0, DIM, DIM) = -pressure * Matrixr::Identity();

  // particles_stresses_gpu[tid] = isotropic_part + tau;

  // problems here
  particles_stresses_gpu[tid] = -pressure * Matrix3r::Identity() + tau;
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_STRESS_UPDATE_MUIJOP(
    Matrix3r *particles_stresses_gpu,
    const Matrixr *particles_velocity_gradients_gpu,
    const Real *particles_masses_gpu, Real *I_gpu,
    const Real *particles_volumes_gpu,
    const Real *particles_volumes_original_gpu,
    const uint8_t *particles_colors_gpu, const bool *particles_is_active_gpu,
    const int num_particles, const Real viscosity, const Real bulk_modulus,
    const Real gamma, const int mat_id) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  stress_update_muijop(particles_stresses_gpu, particles_velocity_gradients_gpu,
                       particles_masses_gpu, I_gpu, particles_volumes_gpu,
                       particles_volumes_original_gpu, particles_colors_gpu,
                       particles_is_active_gpu, viscosity, bulk_modulus, gamma,
                       mat_id, tid);
}

#endif

} // namespace pyroclastmpm