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

/*
[1] https://en.wikipedia.org/wiki/Infinitesimal_strain_theory
[2] de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen. Computational
methods for plasticity: theory and applications. John Wiley & Sons, 2011.
*/

#include "pyroclastmpm/materials/mohrcoulomb.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern const Real dt_cpu;
#endif

/**
 * @brief Update stress using von mises yield criterion
 *
 * @param particles_stresses_gpu  particle stress tensor
 * @param particles_eps_e_gpu  elastic strain tensor
 * @param particles_acc_eps_p_gpu scalar accumulated plastic strain
 * @param particles_velocity_gradient_gpu velocity gradient tensor
 * @param particles_F_gpu deformation gradient tensor
 * @param particle_colors_gpu particle material id
 * @param bulk_modulus bulk modulus
 * @param shear_modulus shear modulus
 * @param yield_stress initial yield stress
 * @param H hardening coefficient
 * @param mat_id material id
 * @param tid thread id
 */
__device__ __host__ inline void
update_vonmises(Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
                Real *particles_acc_eps_p_gpu,
                const Matrixr *particles_velocity_gradient_gpu,
                const uint8_t *particle_colors_gpu, const Real bulk_modulus,
                const Real shear_modulus, const Real yield_stress, const Real H,
                const int mat_id, const bool do_update_history,
                const bool is_velgrad_strain_increment, const int tid) {

  if (particle_colors_gpu[tid] != mat_id) {
    return;
  }
#if DIM == 3
#ifdef CUDA_ENABLED
  const double dt = dt_gpu;
#else
  const double dt = dt_cpu;
#endif

  // Velocity gradient
  Matrixr vel_grad = particles_velocity_gradient_gpu[tid];

  Matrixr deps_curr;
  if (is_velgrad_strain_increment) {
    // only useful for single point integration test
    deps_curr = vel_grad;
  } else {
    const Matrixr vel_grad_T = vel_grad.transpose();
    deps_curr = 0.5 * (vel_grad + vel_grad_T) * dt;
  }
  // strain increment deps (dot eps)*dt

  // elastic strain (previous step)
  const Matrixr eps_e_prev = particles_eps_e_gpu[tid];

  //   // trail elastic strain eq (7.21) [2]
  const Matrixr eps_e_trail = eps_e_prev + deps_curr; // trial elastic strain

  // hydrostatic stress eq (7.82) and volumetric strain eq (3.90) [2]
  const Real eps_e_v_trail = eps_e_trail.trace();

  // pressure eq (7.82) [2]
  const Real p_trail = bulk_modulus * eps_e_v_trail;

  // deviatoric stress (7.82) and strain eq (3.114) [2]
  Matrix3r eps_e_dev_trail;

  // plane strain condition
  eps_e_dev_trail.block(0, 0, DIM, DIM) = eps_e_trail;
  eps_e_dev_trail -= (1 / 3.) * eps_e_v_trail * Matrix3r::Identity();

  const Matrix3r s_trail = 2. * shear_modulus * eps_e_dev_trail;

  // accumulated plastic strain
  const Real acc_eps_p_trail = particles_acc_eps_p_gpu[tid];

  // isotropic linear hardening eq (6.170) [2]
  const Real sigma_y_trail = yield_stress + H * acc_eps_p_trail;

  // yield function eq (6.106) and (6.110) [2]
  const Real q_trail =
      (Real)sqrt(3 * 0.5 * (s_trail * s_trail.transpose()).trace());
  const Real Phi_trail = q_trail - sigma_y_trail;

  // printf("Phi_trail: %f \n", Phi_trail);
  //   // #if DIM == 3
  //   // if stress is in feasible region elastic step eq (7.84)

  if (Phi_trail <= 0) {
    particles_stresses_gpu[tid] = s_trail + p_trail * Matrix3r::Identity();
    if (do_update_history) {
      particles_eps_e_gpu[tid] = eps_e_trail;
      particles_acc_eps_p_gpu[tid] = acc_eps_p_trail;
    }
    return;
  }

  //   // #else

  //   //   printf("VON MISES not supported for 2D at the moment\n");
  //   // #endif

  // otherwise do return mapping - box 7.4 [2]
  // find plastic multiplier dgamma, such that yield function is
  // approximately
  // zero using newton raphson method
  double dgamma = 0.0;

  double Phi_approx;
  double acc_eps;

  const double tol = 1e-7;

  int iter = 0; // debug purposes
  do {
    // simplified implicit return mapping equation (7.91)
    // uses the fact that VM flow vector is purely deviatoric
    // (pressure-independent)
    acc_eps = acc_eps_p_trail + dgamma;

    // isotropic linear hardening eq (6.170) [2]
    const double sigma_y = yield_stress + H * acc_eps;
    Phi_approx = q_trail - 3.0 * shear_modulus * dgamma - sigma_y;

    const double d = -3.0 * shear_modulus - H; // residual of theyield
                                               // function

    dgamma = dgamma - Phi_approx / d;

    // printf("dgamma: %f psi_approx %f iter %d H %f \n", dgamma,
    // psi_approx,iter);
    iter += 1;
  } while (abs(Phi_approx) > tol);

  // printf("acc_eps: %f iter %d H %f yield_stress %f \n", acc_eps, iter, H,
  //  yield_stress);
  const Real p_curr =
      p_trail; // since von mises yield function is an isotropic function

  const Matrixr s_curr =
      (1.0 - (3.0 * shear_modulus * dgamma) / q_trail) * s_trail;

  const Matrixr sigma_curr = s_curr + p_trail * Matrixr::Identity();
  const Matrixr eps_e_curr = s_curr / (2.0 * shear_modulus) +
                             (1. / 3.) * eps_e_v_trail * Matrixr::Identity();

  particles_stresses_gpu[tid] = sigma_curr;
  // printf("update yield stress iter: %d\n", iter);
  if (do_update_history) {
    particles_eps_e_gpu[tid] = eps_e_curr;
    particles_acc_eps_p_gpu[tid] = acc_eps;
  }
#else
  printf("VON MISES not supported for 2D at the moment\n");
#endif
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_STRESS_UPDATE_VONMISES(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
    Real *particles_acc_eps_p_gpu,
    const Matrixr *particles_velocity_gradient_gpu,
    const uint8_t *particle_colors_gpu, const Real bulk_modulus,
    const Real shear_modulus, const Real yield_stress, const Real H,
    const int mat_id, const bool do_update_history,
    const bool is_velgrad_strain_increment, const int num_particles) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_particles) {

    return;
  }

  update_vonmises(particles_stresses_gpu, particles_eps_e_gpu,
                  particles_acc_eps_p_gpu, particles_velocity_gradient_gpu,
                  particle_colors_gpu, bulk_modulus, shear_modulus,
                  yield_stress, H, mat_id, do_update_history,
                  is_velgrad_strain_increment, tid);
}
#endif

} // namespace pyroclastmpm