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

extern const int global_step_cpu;

using Vector2hp = Eigen::Matrix<double, 2, 1>;
using Matrix2hp = Eigen::Matrix<double, 2, 2>;

/**
 * @brief Compute preconsolidation pressure
 * @param specific_volume Specific volume with respect to updated volume
 * @param pc_prev Previous preconsolidation pressure
 * @param lam slope of virgin consolidation line
 * @param kap slope of swelling line
 * @param alpha_next Compressive volumetric plastic strain (next step)
 * @param alpha_prev  Compressive volumetric plastic strain (previous step)
 */
__host__ __device__ inline Real
compute_pc(const Real specific_volume, const Real pc_prev, const Real lam,
           const Real kap, const Real alpha_next, const Real alpha_prev) {

  const Real C = specific_volume / (lam - kap);

  return exp(C * (alpha_next - alpha_prev)) * pc_prev;
  // return pc_prev * (1 + alpha_next * C - alpha_prev * C);
}

/**
 * @brief Compute ellipse radius
 * @param Pc preconsolidation pressure
 * @param beta Parameter related to size of outer diameter of ellipse
 * @param Pt  Tensile yield hydrostatic stress
 */
__host__ __device__ inline Real compute_a(const Real Pc, const Real beta,
                                          const Real Pt) {
  return (Pc + Pt) / (1 + beta);
}

__host__ __device__ inline Real compute_b(const Real p, const Real Pt,
                                          const Real beta, const Real a) {
  Real b = 1;
  if (p < Pt - a) {
    b = beta;
  }
  return b;
}

/**
 * @brief Compute yield function
 * @param p Pressure
 * @param Pt Tensile yield hydrostatic stress
 * @param a Ellipse radius
 * @param q von mises effective stress
 * @param b Parameter related to size of outer diameter of ellipse
 * @param M Slope of critical state line
 */
__host__ __device__ inline Real
compute_yield_function(const Real p, const Real Pt, const Real a, const Real q,
                       const Real b, const Real M) {

  return (1. / (b * b)) * (p - Pt + a) * (p - Pt + a) + (q / M) * (q / M) -
         a * a;
}

/**
 * @brief Compute modified cam clay stress update
 *
 * @param particles_stresses_gpu  Stress tensor
 * @param particles_eps_e_gpu  Elastic strain tensor
 * @param particles_volume_gpu volume of the particles
 * @param particles_alpha_gpu Compressive volumetric plastic strain
 * @param pc_gpu preconsolidation pressure
 * @param particles_velocity_gradient_gpu velocity gradient
 * @param particle_colors_gpu material id
 * @param stress_ref_gpu prescribed reference stress
 * @param bulk_modulus bulk modulus (K)
 * @param shear_modulus shear modulus (G)
 * @param M slope of critical state line
 * @param lam slope of virgin consolidation line
 * @param kap slope of swelling line
 * @param Pc0 initial preconsolidation pressure
 * @param Pt tensile yield hydrostatic stress
 * @param beta parameter related to size of outer diameter of ellipse
 * @param Vs volume of the solid
 * @param mat_id material id
 * @param do_update_history  variable if history should be updated
 * @param is_velgrad_strain_increment variable if the velocity gradient is to be
 * used as a strain increment
 * @param tid thread id (particle id)
 */
__device__ __host__ inline void update_modifiedcamclay(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
    const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
    Real *particles_alpha_gpu, Real *pc_gpu,
    const Matrixr *particles_velocity_gradient_gpu,
    const uint8_t *particle_colors_gpu, const Matrixr *stress_ref_gpu,
    const Real bulk_modulus, const Real shear_modulus, const Real M,
    const Real lam, const Real kap, const Real Pc0, const Real Pt,
    const Real beta, const Real Vs, const int mat_id,
    const bool do_update_history, const bool is_velgrad_strain_increment,
    const int tid) {

  if (particle_colors_gpu[tid] != mat_id) {
    return;
  }

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

  // 0. compute strain increment
  Matrixr vel_grad = particles_velocity_gradient_gpu[tid];

  Matrixr deps_curr;

  if (is_velgrad_strain_increment) {
    deps_curr = vel_grad;
  } else {
    const Matrixr vel_grad_T = vel_grad.transpose();
    deps_curr = 0.5 * (vel_grad + vel_grad_T) * dt;
  }

  const Matrix3r stress_ref = stress_ref_gpu[tid];

  const Real p_ref = (Real)(1.0 / 3.0) * stress_ref.trace();

  const Matrix3r s_ref = stress_ref - p_ref * Matrix3r::Identity();

  // 1. trail elastic step
  // trail elastic strain = previous elastic strain + strain increment
  // eq (7.21) [2]
  const Matrixr eps_e_trail =
      particles_eps_e_gpu[tid] + deps_curr; // trial elastic strain

  // trail elastic volumetric strain volumetric strain eq (3.90) [2]
  const Real eps_e_v_trail = eps_e_trail.trace();

  // trail pressure eq (7.82) [2]
  // compression is negative

  const Real p_trail = p_ref + bulk_modulus * eps_e_v_trail;

  // deviatoric strain eq (3.114) [2]
  Matrix3r eps_e_dev_trail;

  // plane strain condition
  eps_e_dev_trail.block(0, 0, DIM, DIM) = eps_e_trail;
  eps_e_dev_trail -= (1 / 3.) * eps_e_v_trail * Matrix3r::Identity();

  // deviatoric stress eq (7.82) [2]
  const Matrix3r s_trail = s_ref + 2. * shear_modulus * eps_e_dev_trail;

  // von mises effective stress
  const Real q_trail =
      (Real)sqrt(3 * 0.5 * (s_trail * s_trail.transpose()).trace());

  // 2. trail hardening step
  // specific volume with respect to initial volume
  const Real specific_volume = particles_volume_gpu[tid] / Vs;

  const Real specific_volume_original = particles_volume_original_gpu[tid] / Vs;

  // compressive positive plastic volumetric strain (previous step)
  const Real alpha_prev = particles_alpha_gpu[tid];

  const Real pc_prev = pc_gpu[tid];

  // ellipse radius
  const Real a_trail = compute_a(pc_prev, beta, Pt);

  // parameter related to size of outer diameter of ellipse
  const Real b_trail = compute_b(p_trail, Pt, beta, a_trail);

  const Real Phi_trail =
      compute_yield_function(p_trail, Pt, a_trail, q_trail, b_trail, M);

  if (Phi_trail <= (Real)0.0) {

    particles_stresses_gpu[tid] = s_trail + p_trail * Matrix3r::Identity();

    if (do_update_history) {
      particles_eps_e_gpu[tid] = eps_e_trail;
    }
    return;
  }

  Vector2hp R = Vector2hp::Zero();

  // plastic multiplier (0) compressive volumetric plastic strain (1)
  Vector2hp OptVariables = Vector2hp::Zero();
  OptVariables[0] = 0.0;
  OptVariables[1] = alpha_prev;

  // 4. Newton-Raphson iteration
  double p_next;
  double q_next;
  Matrixr s_next;
  double pc_next;
  int counter = 0;
  double conv = 1e10;
  do {

    const double dgamma_next = OptVariables[0];
    const double alpha_next = OptVariables[1];

    p_next = p_trail + bulk_modulus * (alpha_next - alpha_prev);

    q_next = ((M * M) / (M * M + 6 * shear_modulus * dgamma_next)) * q_trail;

    pc_next = compute_pc(specific_volume_original, pc_prev, lam, kap,
                         alpha_next, alpha_prev);

    const double a_next = compute_a((Real)pc_next, beta, Pt);

    const double b_next = compute_b((Real)p_next, Pt, beta, a_next);

    s_next = ((M * M) / (M * M + 6 * shear_modulus * dgamma_next)) * s_trail;

    R[0] = compute_yield_function(p_next, Pt, a_next, q_next, b_next, M);
    R[1] =
        alpha_next - alpha_prev +
        dgamma_next * ((double)2. / (b_next * b_next)) * (p_next - Pt + a_next);

    // 5. compute Jacobian
    // slope of the hardening curve
    // const double dPc = pc_prev * ((specific_volume) / (lam - kap));

    const double dPc = pc_next * ((specific_volume_original) / (lam - kap));

    const double H = dPc / (1 + beta);

    const double p_overline = p_next - Pt + a_next;

    Matrix2hp d = Matrix2hp::Zero();

    d(0, 0) =
        ((-12 * shear_modulus) / (M * M + 6 * shear_modulus * dgamma_next)) *
        (q_next / M) * (q_next / M);

    d(0, 1) = ((2.0 * p_overline) / (b_next * b_next)) * (bulk_modulus + H) -
              2.0 * a_next * H;

    d(1, 0) = (2.0 * p_overline) / (b_next * b_next);

    d(1, 1) =
        1 + ((2.0 * dgamma_next) / (b_next * b_next)) * (bulk_modulus + H);

    const Matrix2hp d_inv = d.inverse();
    OptVariables = OptVariables - d_inv * R;

    conv = R.norm();

    counter++;
  } while (conv > 1e-4);

  const Matrixr sigma_next = s_next + p_next * Matrix3r::Identity();

  const Matrixr eps_e_curr =
      (s_next - s_ref) / (2.0 * shear_modulus) +
      (p_next - p_ref) / (3. * bulk_modulus) * Matrixr::Identity();

  particles_stresses_gpu[tid] = sigma_next;

  if (do_update_history) {
    particles_eps_e_gpu[tid] = eps_e_curr;

    particles_alpha_gpu[tid] = OptVariables[1];
    pc_gpu[tid] = pc_next;
  }
}

} // namespace pyroclastmpm