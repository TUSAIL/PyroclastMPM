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
compute_yield_function_cap(const Real p, const Real Pt, const Real a,
                           const Real q, const Real b, const Real M) {

  return (1. / (b * b)) * (p - Pt + a) * (p - Pt + a) + (q / M) * (q / M) -
         a * a;
}

__host__ __device__ inline Real compute_yield_function_cone(const Real p,
                                                            const Real Pt,
                                                            const Real J2,
                                                            const Real eta) {

  // return (1. / (b * b)) * (p - Pt + a) * (p - Pt + a) + (q / M) * (q / M) -
  //  a * a;

  return sqrt(J2) + eta * (p - Pt);
}

__host__ __device__ inline void
return_mapping_cone(double &p_next, Matrix3r &s_next, double &dgamma_a_next,
                    double &alpha_next, const Matrix3r s_trail,
                    const double J2_trail, const double p_trail,
                    const Real eta_overline, const Real eta,
                    const double alpha_prev, const double bulk_modulus,
                    const double shear_modulus, const double Phi_cone_trail) {
  // closed form solution since cone is perfectly plastic
  dgamma_a_next =
      Phi_cone_trail / (bulk_modulus * eta * eta_overline + shear_modulus);

  alpha_next = alpha_prev - eta_overline * dgamma_a_next;

  p_next = p_trail - bulk_modulus * eta_overline * dgamma_a_next;

  s_next = (1.0 - (shear_modulus * dgamma_a_next) / sqrt(J2_trail)) * s_trail;
}

__host__ __device__ inline void return_mapping_apex(
    double &p_next, Matrix3r &s_next, Real &dgamma_a_next, double &alpha_next,
    const double p_trail, const Real eta_overline, const Real eta,
    const double alpha_prev, const double Pt, const double bulk_modulus) {

  dgamma_a_next = (p_trail - Pt) / (eta_overline * bulk_modulus);

  alpha_next = alpha_prev - eta_overline * dgamma_a_next;

  p_next = p_trail - bulk_modulus * eta_overline * dgamma_a_next;

  s_next = Matrix3r::Zero();
}

//  output s_next, p_next, alpha_next, pc_next
__host__ __device__ inline void return_mapping_cap(
    double &p_next, double &q_next, Matrix3r &s_next, double &pc_next,
    double &alpha_next, double &a_next, const Matrix3r s_trail,
    const double p_trail, const double q_trail, const double pc_prev,
    const double alpha_prev, const double Pt, const double bulk_modulus,
    const double shear_modulus, const double beta, const double M,
    const double lam, const double kap, const double specific_volume_original

) {

  Vector2hp R = Vector2hp::Zero();

  // plastic multiplier (0) compressive volumetric plastic strain (1)
  Vector2hp OptVariables = Vector2hp::Zero();
  OptVariables[0] = 0.0;
  OptVariables[1] = alpha_prev;

  int counter = 0;
  double conv = 1e10;
  double tol = 1e-1;

  do {

    const double dgamma_next = OptVariables[0];
    // const double alpha_next = OptVariables[1];
    alpha_next = OptVariables[1];
    p_next = p_trail + bulk_modulus * (alpha_next - alpha_prev);

    q_next = ((M * M) / (M * M + 6 * shear_modulus * dgamma_next)) * q_trail;

    pc_next = compute_pc(specific_volume_original, pc_prev, lam, kap,
                         alpha_next, alpha_prev);

    a_next = compute_a((Real)pc_next, beta, Pt);

    const double b_next = compute_b((Real)p_next, Pt, beta, a_next);

    s_next = ((M * M) / (M * M + 6 * shear_modulus * dgamma_next)) * s_trail;

    R[0] = compute_yield_function_cap(p_next, Pt, a_next, q_next, b_next, M);
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
    if (counter > 1000) {
      tol *= 10;
      // printf("counter %d cov %.16f \n increasing tol %.16f \n", counter,
      // conv,
      //        tol);
    }
  } while (conv > tol);
}

//  output s_next, p_next, alpha_next, pc_next
__host__ __device__ inline void return_mapping_cap_cone(
    double &p_next, double &q_next, Matrix3r &s_next, double &pc_next,
    double &alpha_next, double &a_next, double &dgamma_a_next,
    double &dgamma_b_next, const Matrix3r s_trail, const double p_trail,
    const double q_trail, const double J2_trail, const double pc_prev,
    const double alpha_prev, const double Pt, const double bulk_modulus,
    const double shear_modulus, const double beta, const double M,
    const double lam, const double kap, const double specific_volume_original,
    const double eta_overline, const double eta

) {

  Vector2hp R = Vector2hp::Zero();

  // plastic multiplier (0) compressive volumetric plastic strain (1)
  Vector2hp OptVariables = Vector2hp::Zero();
  OptVariables[0] = 0.0;
  OptVariables[1] = 0.0;

  int counter = 0;
  double conv = 1e10;
  double tol = 1e-1;

  do {
    dgamma_a_next = OptVariables[0];
    dgamma_b_next = OptVariables[1];

    p_next = p_trail - bulk_modulus * eta_overline * dgamma_a_next;

    // for cone
    const double sqrt_J2_recip =
        (1.0 + (6.0 * shear_modulus * dgamma_b_next) / pow(M, 2));

    const double sqrt_J2_next =
        (sqrt(J2_trail) - shear_modulus * dgamma_a_next) / sqrt_J2_recip;

    // for cap
    const double q_next = sqrt(3) * sqrt_J2_next;

    alpha_next = alpha_prev - eta_overline * dgamma_a_next;

    pc_next = compute_pc(specific_volume_original, pc_prev, lam, kap,
                         alpha_next, alpha_prev);

    a_next = compute_a((Real)pc_next, beta, Pt);

    R[0] = sqrt_J2_recip + eta * (p_next - Pt); // DP yield function
    R[1] = compute_yield_function_cap(p_next, Pt, a_next, q_next, beta,
                                      M); // MCC yield function

    // cone derivatives for jacobian
    const double dp_next_dgamma_a_next = -bulk_modulus * eta_overline;
    const double dsqrt_J2_next_dgamma_a_next = -(shear_modulus / sqrt_J2_recip);

    const double dsqrt_J2_next_dgamma_b_next =
        (-sqrt_J2_next / sqrt_J2_recip) * (6.0 * shear_modulus / pow(M, 2));

    // cap derivatives for jacobian
    const double dq_next_dgamma_a_next = sqrt(3) * dsqrt_J2_next_dgamma_a_next;
    const double dq_next_dgamma_b_next = sqrt(3) * dsqrt_J2_next_dgamma_b_next;

    const double da_next_dgamma_a_next =
        -(1.0 / (1.0 + beta)) * (specific_volume_original / (lam - kap)) *
        eta_overline * pc_next;

    Matrix2hp d = Matrix2hp::Zero();

    d(0, 0) = dsqrt_J2_next_dgamma_a_next + eta * dp_next_dgamma_a_next;

    d(0, 1) = dsqrt_J2_next_dgamma_b_next;

    d(1, 0) = (2.0 / beta * beta) * (p_next - Pt + a_next) *
                  (dp_next_dgamma_a_next + da_next_dgamma_a_next) +
              2 * (q_next / (M * M)) * dq_next_dgamma_a_next -
              2.0 * alpha_next * da_next_dgamma_a_next;

    d(1, 1) = 2.0 * (q_next / (M * M)) * dq_next_dgamma_b_next;

    const Matrix2hp d_inv = d.inverse();
    OptVariables = OptVariables - d_inv * R;

    conv = R.norm();

    counter++;
    if (counter > 1000) {
      tol *= 10;
      // printf("counter %d cov %.16f \n increasing tol %.16f \n", counter,
      // conv,
      //        tol);
    }

  } while (conv > tol);
}

__device__ __host__ inline void
update_plasticity(const Matrix3r s_next, const double p_next, double alpha_next,
                  double pc_next, const Matrix3r s_ref, double p_ref,
                  const double shear_modulus, const double bulk_modulus,
                  bool do_update_history, Matrix3r *particles_stresses_gpu,
                  Matrixr *particles_eps_e_gpu, Real *particles_alpha_gpu,
                  Real *pc_gpu, int tid) {

  const Matrix3r eps_e_curr_3D =
      (s_next - s_ref) / (2.0 * shear_modulus) +
      (p_next - p_ref) / (3. * bulk_modulus) * Matrix3r::Identity();

  particles_stresses_gpu[tid] = s_next + p_next * Matrix3r::Identity();

  if (do_update_history) {
    particles_eps_e_gpu[tid] = eps_e_curr_3D.block(0, 0, DIM, DIM);

    particles_alpha_gpu[tid] = alpha_next;
    pc_gpu[tid] = pc_next;
  }
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
 * @param Pt tensile yield hydrostatic stress
 * @param beta parameter related to size of outer diameter of ellipse
 * @param Vs volume of the solid
 * @param mat_id material id
 * @param do_update_history  variable if history should be updated
 * @param is_velgrad_strain_increment variable if the velocity gradient is
 * to be used as a strain increment
 * @param tid thread id (particle id)
 */
__device__ __host__ inline void update_druckerpragercap(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
    const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
    Real *particles_alpha_gpu, Real *pc_gpu,
    const Matrixr *particles_velocity_gradient_gpu,
    const uint8_t *particle_colors_gpu, const Matrix3r *stress_ref_gpu,
    const Real bulk_modulus, const Real shear_modulus, const Real M,
    const Real lam, const Real kap, const Real Pt, const Real beta,
    const Real Vs, const Real eta_overline, const int mat_id,
    const bool do_update_history, const bool is_velgrad_strain_increment,
    const int tid) {

  // #if DIM > 2 // TODO FIX THIS
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

    Matrixr W = 0.5 * (vel_grad - vel_grad_T) * dt;
    const Matrixr eps_e_prev_gou = particles_eps_e_gpu[tid];
    deps_curr += -W * eps_e_prev_gou + eps_e_prev_gou * W;
  }

  const Matrix3r stress_ref = stress_ref_gpu[tid];

  const Real p_ref = (Real)(1.0 / 3.0) * stress_ref.trace();

  const Matrix3r s_ref = stress_ref - p_ref * Matrix3r::Identity();

  // 1. trail elastic step
  // trail elastic strain = previous elastic strain + strain increment
  const Matrixr eps_e_trail =
      particles_eps_e_gpu[tid] + deps_curr; // trial elastic strain

  // trail elastic volumetric strain volumetric strain
  const Real eps_e_v_trail = eps_e_trail.trace();

  // trail pressure
  // compression is negative
  const Real p_trail = p_ref + bulk_modulus * eps_e_v_trail;

  // deviatoric strain eq (3.114) [2]
  // compatible with plane strain
  Matrix3r eps_e_dev_trail;
  eps_e_dev_trail.block(0, 0, DIM, DIM) = eps_e_trail;
  eps_e_dev_trail -= (1 / 3.) * eps_e_v_trail * Matrix3r::Identity();
#if DIM < 3
  eps_e_dev_trail(2, 2) = 0;
#endif

  // deviatoric stress
  const Matrix3r s_trail = s_ref + 2. * shear_modulus * eps_e_dev_trail;

  const Real J2_trail = 0.5 * (s_trail * s_trail.transpose()).trace();

  // von mises effective stress
  const Real q_trail = (Real)sqrt(3 * J2_trail);

  /// start DP cap
  // const Real eta = sqrt(2. / 3.) * M; // textbook is wrong
  const Real eta = sqrt(1. / 3.) * M;
  const Real Phi_cone_trail =
      compute_yield_function_cone(p_trail, Pt, J2_trail, eta);

  // 2. trail hardening step
  // specific volume with respect to initial volume
  const Real specific_volume_original = particles_volume_original_gpu[tid] / Vs;

  // compressive positive plastic volumetric strain (previous step)
  const Real alpha_prev = particles_alpha_gpu[tid];

  const Real pc_prev = pc_gpu[tid];

  // ellipse radius
  const Real a_trail = compute_a(pc_prev, beta, Pt);

  // parameter related to size of outer diameter of ellipse
  const Real b_trail = compute_b(p_trail, Pt, beta, a_trail);

  const Real Phi_cap_trail =
      compute_yield_function_cap(p_trail, Pt, a_trail, q_trail, b_trail, M);

  // check if yield functions are within elastic domain

  if ((Phi_cone_trail <= (Real)0.0) && (Phi_cap_trail <= (Real)0.0)) {
    particles_stresses_gpu[tid] = s_trail + p_trail * Matrix3r::Identity();

    if (do_update_history) {
      particles_eps_e_gpu[tid] = eps_e_trail;
    }
    return;
  }

  // 4. Newton-Raphson iteration
  double p_next = p_trail;
  Matrix3r s_next = s_trail;

  double q_next = q_trail;

  double pc_next = pc_prev;

  double a_next = a_trail;

  double alpha_next = alpha_prev;

  double dgamma_a_next = 0.0;
  double dgamma_b_next = 0.0;

  // check if yield functions is within cone
  if (Phi_cone_trail > (Real)0.0) {

    return_mapping_cone(p_next, s_next, dgamma_a_next, alpha_next, s_trail,
                        J2_trail, p_trail, eta_overline, eta, alpha_prev,
                        bulk_modulus, shear_modulus, Phi_cone_trail);

    const bool is_apex =
        sqrt(J2_trail) - shear_modulus * dgamma_a_next < (Real)0.0;

    // check if we project bact to apex
    if (is_apex) {

      return_mapping_apex(p_next, s_next, dgamma_a_next, alpha_next, p_trail,
                          eta_overline, eta, alpha_prev, Pt, bulk_modulus);

      pc_next = compute_pc(specific_volume_original, pc_prev, lam, kap,
                           alpha_next, alpha_prev);

      update_plasticity(s_next, p_next, alpha_next, pc_next, s_ref, p_ref,
                        shear_modulus, bulk_modulus, do_update_history,
                        particles_stresses_gpu, particles_eps_e_gpu,
                        particles_alpha_gpu, pc_gpu, tid);
      return;
    }

    pc_next = compute_pc(specific_volume_original, pc_prev, lam, kap,
                         alpha_next, alpha_prev);

    a_next = compute_a((Real)pc_next, beta, Pt);
  }

  if ((Phi_cap_trail > (Real)0.0) && (p_next < a_next)) {
    // update pc and a_next included in return_mapping_cap
    return_mapping_cap(p_next, q_next, s_next, pc_next, alpha_next, a_next,
                       s_trail, p_trail, q_trail, pc_prev, alpha_prev, Pt,
                       bulk_modulus, shear_modulus, beta, M, lam, kap,
                       specific_volume_original);

    if (p_next > -a_next) {

      // return_mapping_cap_cone(p_next, q_next, s_next, pc_next, alpha_next,
      //                         a_next, dgamma_a_next, dgamma_b_next, s_trail,
      //                         p_trail, q_trail, J2_trail, pc_prev,
      //                         alpha_prev, Pt, bulk_modulus, shear_modulus,
      //                         beta, M, lam, kap, specific_volume_original,
      //                         eta_overline, eta);
      // // update pc and a_next included in return_mapping_cap_cone
      // update_plasticity(s_next, p_next, alpha_next, pc_next, s_ref, p_ref,
      //                   shear_modulus, bulk_modulus, do_update_history,
      //                   particles_stresses_gpu, particles_eps_e_gpu,
      //                   particles_alpha_gpu, pc_gpu, tid);
      // return;
    }
  }

  update_plasticity(s_next, p_next, alpha_next, pc_next, s_ref, p_ref,
                    shear_modulus, bulk_modulus, do_update_history,
                    particles_stresses_gpu, particles_eps_e_gpu,
                    particles_alpha_gpu, pc_gpu, tid);
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_STRESS_UPDATE_DPCAP(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
    const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
    Real *particles_alpha_gpu, Real *pc_gpu,
    const Matrixr *particles_velocity_gradient_gpu,
    const uint8_t *particle_colors_gpu, const Matrix3r *stress_ref_gpu,
    const Real bulk_modulus, const Real shear_modulus, const Real M,
    const Real lam, const Real kap, const Real Pt, const Real beta,
    const Real Vs,const Real eta_overline, const int mat_id, const bool do_update_history,
    const bool is_velgrad_strain_increment, const int num_particles) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads


  update_druckerpragercap(
    particles_stresses_gpu,
    particles_eps_e_gpu,
    particles_volume_gpu,
    particles_volume_original_gpu,
    particles_alpha_gpu,
    pc_gpu,
    particles_velocity_gradient_gpu,
    particle_colors_gpu,
    stress_ref_gpu,
    bulk_modulus,
    shear_modulus,
    M,
    lam,
    kap,
    Pt,
    beta,
    Vs,
    eta_overline,
    mat_id,
    do_update_history,
    is_velgrad_strain_increment,
    tid);

}
#endif

} // namespace pyroclastmpm