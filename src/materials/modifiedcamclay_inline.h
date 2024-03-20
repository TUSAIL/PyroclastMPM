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

namespace pyroclastmpm
{

#ifdef CUDA_ENABLED
  extern Real __constant__ dt_gpu;
#else
  extern const Real dt_cpu;
#endif

  extern const int global_step_cpu;

  using Vector2hp = Eigen::Matrix<double, 2, 1>;
  using Matrix2hp = Eigen::Matrix<double, 2, 2>;

  /**
   * @brief Compute yield function
   * @param p Pressure
   * @param Pt Tensile yield hydrostatic stress
   * @param p_s Back stress
   * @param q von mises effective stress
   * @param b Parameter related to size of outer diameter of ellipse
   * @param M Slope of critical state line
   */
  __host__ __device__ inline Real
  compute_yield_function(const Real p, const Real Pt, const Real p_s, const Real q, const Real M)
  {

    return (p_s - p - Pt) * (p_s - p - Pt) + (q / M) * (q / M) -
           p_s * p_s;
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
   * @param is_velgrad_strain_increment variable if the velocity gradient is to be
   * used as a strain increment
   * @param tid thread id (particle id)
   */
  __device__ __host__ inline void update_modifiedcamclay(
      Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
      const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
      Real *particles_alpha_gpu, Real *pc_gpu,
      const Matrixr *particles_velocity_gradient_gpu,
      const uint8_t *particle_colors_gpu, const Matrix3r *stress_ref_gpu,
      const Real bulk_modulus, const Real shear_modulus, const Real M,
      const Real lam, const Real kap, const Real Pt, const Real beta,
      const Real Vs, const int mat_id, const bool do_update_history,
      const bool is_velgrad_strain_increment, const int tid)
  {

    // #if DIM > 2 // TODO FIX THIS
    if (particle_colors_gpu[tid] != mat_id)
    {
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

    if (is_velgrad_strain_increment) // for servo control
    {
      deps_curr = vel_grad;
    }
    else
    {
      const Matrixr vel_grad_T = vel_grad.transpose();
      deps_curr = 0.5 * (vel_grad + vel_grad_T) * dt;

      Matrixr W = 0.5 * (vel_grad - vel_grad_T) * dt;
      const Matrixr eps_e_prev_gou = particles_eps_e_gpu[tid];
      deps_curr += -W * eps_e_prev_gou + eps_e_prev_gou * W;
    }

    const Matrix3r stress_ref = stress_ref_gpu[tid];
    Real p_ref = -(Real)(1.0 / 3.0) * stress_ref.trace(); // replace by DIM?

    const Matrix3r s_ref = stress_ref + p_ref * Matrix3r::Identity();

    // 1. Trail elastic step

    // strains
    const Matrixr eps_e_trail =
        particles_eps_e_gpu[tid] + deps_curr;        // trail elastic strain
    const Real eps_e_v_trail = -eps_e_trail.trace(); // trail elastic volumetric strain

    // deviatoric strain (work around for 2D plane strain condition)
    Matrix3r eps_e_dev_trail = Matrix3r::Zero();
    eps_e_dev_trail.block(0, 0, DIM, DIM) = eps_e_trail;
    eps_e_dev_trail += (1.0 / 3.0) * eps_e_v_trail * Matrix3r::Identity(); // replace by DIM?
#if DIM < 3
    eps_e_dev_trail(2, 2) = 0;
#endif

    // stresses
    const Real p_trail = p_ref + bulk_modulus * eps_e_v_trail; // TODO reference change sign
    const Matrix3r s_trail = s_ref + 2. * shear_modulus * eps_e_dev_trail;
    const Real q_trail =
        (Real)sqrt(3 * 0.5 * (s_trail * s_trail.transpose()).trace()); // von mises effective stress

    const Real specific_volume = particles_volume_gpu[tid] / Vs;

    const Real p_c_prev = pc_gpu[tid];

    // back stress
    const Real p_s_trail = 0.5 * (p_c_prev + Pt);

    const Real f_trail =
        compute_yield_function(p_trail, Pt, p_s_trail, q_trail, M);

    if (f_trail <= (Real)0.0)
    {

      particles_stresses_gpu[tid] = s_trail - p_trail * Matrix3r::Identity();

      if (do_update_history)
      {
        particles_eps_e_gpu[tid] = eps_e_trail;
      }
      return;
    }

    Vector2hp R = Vector2hp::Zero();
    const Real eps_p_v_prev = particles_alpha_gpu[tid];

    // plastic multiplier (0) volumetric plastic strain (1)
    Vector2hp Solution = Vector2hp::Zero();
    Solution[0] = 0.0;
    Solution[1] = eps_p_v_prev; // total volumetric plastic strain

    // complementary equations
    double p_next = 0.0;
    double q_next = 0.0;
    double p_s_next;
    Matrix3r s_next;
    double tol = 1e-2;
    int counter = 0;
    double conv = 1e10;

    double const v_lam_tilde = specific_volume / (lam - kap);

    // 4. Newton-Raphson iteration
    do
    {

      const double plastic_multi_next = Solution[0];
      const double eps_p_v_next = Solution[1];

      const Real deps_p_v = eps_p_v_next - eps_p_v_prev;

      p_next = p_trail - bulk_modulus * deps_p_v;

      q_next = ((M * M) / (M * M + 6 * shear_modulus * plastic_multi_next)) * q_trail;

      p_s_next = 0.5 * (p_c_prev * (1.0 + v_lam_tilde * deps_p_v) + Pt); // backstress

      s_next = s_next = ((M * M) / (M * M + 6.0 * shear_modulus * Solution[0])) * s_trail;

      R[0] = compute_yield_function(p_next, Pt, p_s_next, q_next, M);
      R[1] =
          eps_p_v_next - eps_p_v_prev +
          2.0 * plastic_multi_next * (p_s_next - p_next - Pt);

      const Real K = -bulk_modulus; // partial of p with respect to plastic volumetric strain

      const double H = 0.5 * v_lam_tilde * (p_c_prev / pow(1.0 - v_lam_tilde * deps_p_v, 2)); // partial of p_c with respect to plastic volumetric strain

      const double p_overline = p_s_next - p_next - Pt;

      Matrix2hp d = Matrix2hp::Zero();

      d(0, 0) =
          ((-12 * shear_modulus) / (M * M + 6 * shear_modulus * plastic_multi_next)) *
          (q_next / M) * (q_next / M);

      d(0, 1) = (2.0 * p_overline) * (bulk_modulus + H) -
                2.0 * p_s_next * H;

      d(1, 0) = 2.0 * p_overline;

      d(1, 1) =
          1.0 + (2.0 * plastic_multi_next) * (bulk_modulus + H);

      const Matrix2hp d_inv = d.inverse();
      Solution = Solution - d_inv * R;

      conv = R.norm();

      // if (counter > 1000)
      // {
      //   // tol *= 10;
      // }
    } while (abs(conv) > tol);
    Matrix3r sigma_next = s_next - p_next * Matrix3r::Identity();

    const Matrix3r eps_e_curr_3D =
        (s_next - s_ref) / (2.0 * shear_modulus) -
        (p_next - p_ref) / (3. * bulk_modulus) * Matrix3r::Identity();

    particles_stresses_gpu[tid] = sigma_next;

    if (do_update_history)
    {
      particles_eps_e_gpu[tid] = eps_e_curr_3D.block(0, 0, DIM, DIM);

      particles_alpha_gpu[tid] = Solution[1];
      pc_gpu[tid] = 2.0 * p_s_next - Pt;
    }
  }

#ifdef CUDA_ENABLED
  __global__ void KERNEL_STRESS_UPDATE_MCC(
      Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
      const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
      Real *particles_alpha_gpu, Real *pc_gpu,
      const Matrixr *particles_velocity_gradient_gpu,
      const uint8_t *particle_colors_gpu, const Matrix3r *stress_ref_gpu,
      const Real bulk_modulus, const Real shear_modulus, const Real M,
      const Real lam, const Real kap, const Real Pt, const Real beta,
      const Real Vs, const int mat_id, const bool do_update_history,
      const bool is_velgrad_strain_increment, const int num_particles)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_particles)
    {
      return;
    } // block access threads

    update_modifiedcamclay(
        particles_stresses_gpu, particles_eps_e_gpu, particles_volume_gpu,
        particles_volume_original_gpu, particles_alpha_gpu, pc_gpu,
        particles_velocity_gradient_gpu, particle_colors_gpu, stress_ref_gpu,
        bulk_modulus, shear_modulus, M, lam, kap, Pt, beta, Vs, mat_id,
        do_update_history, is_velgrad_strain_increment, tid);
  }
#endif

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
   * @param is_velgrad_strain_increment variable if the velocity gradient is to be
   * used as a strain increment
   * @param tid thread id (particle id)
   */
  __device__ __host__ inline void update_modifiedcamclay_nonlinear(
      Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
      const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
      Real *particles_alpha_gpu, Real *pc_gpu,
      const Matrixr *particles_velocity_gradient_gpu,
      const uint8_t *particle_colors_gpu, const Matrix3r *stress_ref_gpu,
      const Real bulk_modulus, const Real shear_modulus, const Real M,
      const Real lam, const Real kap, const Real Pt, const Real beta,
      const Real Vs, const int mat_id, const bool do_update_history,
      const bool is_velgrad_strain_increment, const int tid)
  {

    // #if DIM > 2 // TODO FIX THIS
    if (particle_colors_gpu[tid] != mat_id)
    {
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

    if (is_velgrad_strain_increment) // for servo control
    {
      deps_curr = vel_grad;
    }
    else
    {
      const Matrixr vel_grad_T = vel_grad.transpose();
      deps_curr = 0.5 * (vel_grad + vel_grad_T) * dt;

      Matrixr W = 0.5 * (vel_grad - vel_grad_T) * dt;
      const Matrixr eps_e_prev_gou = particles_eps_e_gpu[tid];
      deps_curr += -W * eps_e_prev_gou + eps_e_prev_gou * W;
    }

    const Matrix3r stress_ref = stress_ref_gpu[tid];
    Real p_ref = -(Real)(1.0 / 3.0) * stress_ref.trace(); // replace by DIM?

    const Matrix3r s_ref = stress_ref + p_ref * Matrix3r::Identity();

    // 1. Trail elastic step

    // strains
    const Matrixr eps_e_trail =
        particles_eps_e_gpu[tid] + deps_curr;        // trail elastic strain
    const Real eps_e_v_trail = -eps_e_trail.trace(); // trail elastic volumetric strain

    // deviatoric strain (work around for 2D plane strain condition)
    Matrix3r eps_e_dev_trail = Matrix3r::Zero();
    eps_e_dev_trail.block(0, 0, DIM, DIM) = eps_e_trail;
    eps_e_dev_trail += (1.0 / 3.0) * eps_e_v_trail * Matrix3r::Identity(); // replace by DIM?
#if DIM < 3
    eps_e_dev_trail(2, 2) = 0;
#endif
    const Real specific_volume = particles_volume_gpu[tid] / Vs;

    // stresses
    // const Real p_trail = p_ref + bulk_modulus * eps_e_v_trail; // TODO reference change sign
    const Real p_prev = -particles_stresses_gpu[tid].trace() / 3.0;

    const Real v_kap = specific_volume / kap;
    const Real deps_e_v_trail = -deps_curr.trace();

    const Real p_trail = p_prev / (1.0 - v_kap * deps_e_v_trail);

    const Matrix3r s_trail = s_ref + 2. * shear_modulus * eps_e_dev_trail;
    const Real q_trail =
        (Real)sqrt(3 * 0.5 * (s_trail * s_trail.transpose()).trace()); // von mises effective stress

    const Real p_c_prev = pc_gpu[tid];

    // back stress
    const Real p_s_trail = 0.5 * (p_c_prev + Pt);

    const Real f_trail =
        compute_yield_function(p_trail, Pt, p_s_trail, q_trail, M);

    if (f_trail <= (Real)0.0)
    {

      particles_stresses_gpu[tid] = s_trail - p_trail * Matrix3r::Identity();

      if (do_update_history)
      {
        particles_eps_e_gpu[tid] = eps_e_trail;
      }
      return;
    }

    Vector2hp R = Vector2hp::Zero();
    const Real eps_p_v_prev = particles_alpha_gpu[tid];

    // plastic multiplier (0) volumetric plastic strain (1)
    Vector2hp Solution = Vector2hp::Zero();
    Solution[0] = 0.0;
    Solution[1] = eps_p_v_prev; // total volumetric plastic strain

    // complementary equations
    double p_next = 0.0;
    double q_next = 0.0;
    double p_s_next;
    Matrix3r s_next;
    double tol = 1e-2;
    int counter = 0;
    double conv = 1e10;

    Real deps_e_v = 0.0;

    double const v_lam_tilde = specific_volume / (lam - kap);

    // 4. Newton-Raphson iteration
    do
    {

      const double plastic_multi_next = Solution[0];
      const double eps_p_v_next = Solution[1];

      const Real deps_p_v = eps_p_v_next - eps_p_v_prev;

      deps_e_v = deps_e_v_trail - deps_p_v;

      p_next = p_prev / (1.0 - v_kap * deps_e_v);

      q_next = ((M * M) / (M * M + 6 * shear_modulus * plastic_multi_next)) * q_trail;

      p_s_next = 0.5 * (p_c_prev * (1.0 + v_lam_tilde * deps_p_v) + Pt); // backstress

      s_next = s_next = ((M * M) / (M * M + 6.0 * shear_modulus * Solution[0])) * s_trail;

      R[0] = compute_yield_function(p_next, Pt, p_s_next, q_next, M);
      R[1] =
          eps_p_v_next - eps_p_v_prev +
          2.0 * plastic_multi_next * (p_s_next - p_next - Pt);

      // const Real K = -bulk_modulus; // partial of p with respect to plastic volumetric strain

      const Real K = -v_kap * p_next * p_next * (1.0 / p_prev);

      const double H = 0.5 * v_lam_tilde * (p_c_prev / pow(1.0 - v_lam_tilde * deps_p_v, 2)); // partial of p_c with respect to plastic volumetric strain

      const double p_overline = p_s_next - p_next - Pt;

      Matrix2hp d = Matrix2hp::Zero();

      d(0, 0) =
          ((-12 * shear_modulus) / (M * M + 6 * shear_modulus * plastic_multi_next)) *
          (q_next / M) * (q_next / M);

      d(0, 1) = (2.0 * p_overline) * (bulk_modulus + H) -
                2.0 * p_s_next * H;

      d(1, 0) = 2.0 * p_overline;

      d(1, 1) =
          1.0 + (2.0 * plastic_multi_next) * (bulk_modulus + H);

      const Matrix2hp d_inv = d.inverse();
      Solution = Solution - d_inv * R;

      conv = R.norm();
      if (counter > 1000)
      {
        tol *= 10;
      }
      counter += 1;
    } while (abs(conv) > tol);
    Matrix3r sigma_next = s_next - p_next * Matrix3r::Identity();

    // const Real _bulk_modulus = v_kap * p_next;

    const Matrix3r eps_e_curr_3D =
        (s_next - s_ref) / (2.0 * shear_modulus) - (1. / 3.) * (particles_eps_e_gpu[tid].trace() + deps_e_v) * Matrix3r::Identity();

    particles_stresses_gpu[tid] = sigma_next;

    if (do_update_history)
    {
      particles_eps_e_gpu[tid] = eps_e_curr_3D.block(0, 0, DIM, DIM);

      particles_alpha_gpu[tid] = Solution[1];
      pc_gpu[tid] = 2.0 * p_s_next - Pt;
    }
  }

#ifdef CUDA_ENABLED
  __global__ void KERNEL_STRESS_UPDATE_MCC_NONLINEAR(
      Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
      const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
      Real *particles_alpha_gpu, Real *pc_gpu,
      const Matrixr *particles_velocity_gradient_gpu,
      const uint8_t *particle_colors_gpu, const Matrix3r *stress_ref_gpu,
      const Real bulk_modulus, const Real shear_modulus, const Real M,
      const Real lam, const Real kap, const Real Pt, const Real beta,
      const Real Vs, const int mat_id, const bool do_update_history,
      const bool is_velgrad_strain_increment, const int num_particles)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_particles)
    {
      return;
    } // block access threads

    update_modifiedcamclay_nonlinear(
        particles_stresses_gpu, particles_eps_e_gpu, particles_volume_gpu,
        particles_volume_original_gpu, particles_alpha_gpu, pc_gpu,
        particles_velocity_gradient_gpu, particle_colors_gpu, stress_ref_gpu,
        bulk_modulus, shear_modulus, M, lam, kap, Pt, beta, Vs, mat_id,
        do_update_history, is_velgrad_strain_increment, tid);
  }
#endif

} // namespace pyroclastmpm