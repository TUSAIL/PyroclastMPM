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

#include <cmath>
// using namespace std;

__host__ __device__ inline Real compute_yield_function_cone(const Real p,
                                                            const Real Pt,
                                                            const Real J2,
                                                            const Real eta) {
  return sqrt(J2) + eta * (p - Pt);
}

__host__ __device__ inline void
return_mapping_cone(double &p_next, Matrix3r &s_next, double &dgamma_a_next,
                    const Matrix3r s_trail,
                    const double J2_trail, const double p_trail,
                    const Real eta_overline, const Real eta,
                    const double bulk_modulus,
                    const double shear_modulus, const double Phi_cone_trail) {
  // closed form solution since cone is perfectly plastic
  dgamma_a_next =
      Phi_cone_trail / (bulk_modulus * eta * eta_overline + shear_modulus);


  p_next = p_trail - bulk_modulus * eta_overline * dgamma_a_next;

  s_next = (1.0 - (shear_modulus * dgamma_a_next) / sqrt(J2_trail)) * s_trail;
}

__host__ __device__ inline void return_mapping_apex(
    double &p_next, Matrix3r &s_next, Real &dgamma_a_next,
    const double p_trail, const Real eta_overline, const Real eta,
    const double Pt, const double bulk_modulus) {

  dgamma_a_next = (p_trail - Pt) / (eta_overline * bulk_modulus);


  p_next = p_trail - bulk_modulus * eta_overline * dgamma_a_next;

  s_next = Matrix3r::Zero();
}

__device__ __host__ inline void
update_plasticity(const Matrix3r s_next, const double p_next,
                  const Matrix3r s_ref, double p_ref,
                  const double shear_modulus, const double bulk_modulus,
                  bool do_update_history, Matrix3r *particles_stresses_gpu,
                  Matrixr *particles_eps_e_gpu, int tid) {

  const Matrix3r eps_e_curr_3D =
      (s_next - s_ref) / (2.0 * shear_modulus) +
      (p_next - p_ref) / (3. * bulk_modulus) * Matrix3r::Identity();

  particles_stresses_gpu[tid] = s_next + p_next * Matrix3r::Identity();

  if (do_update_history) {
    particles_eps_e_gpu[tid] = eps_e_curr_3D.block(0, 0, DIM, DIM);
  }
}

// /**
//  * @brief Compute modified cam clay stress update
//  *
//  * @param particles_stresses_gpu  Stress tensor
//  * @param particles_eps_e_gpu  Elastic strain tensor
//  * @param particles_volume_gpu volume of the particles
//  * @param particles_alpha_gpu Compressive volumetric plastic strain
//  * @param pc_gpu preconsolidation pressure
//  * @param particles_velocity_gradient_gpu velocity gradient
//  * @param particle_colors_gpu material id
//  * @param stress_ref_gpu prescribed reference stress
//  * @param bulk_modulus bulk modulus (K)
//  * @param shear_modulus shear modulus (G)
//  * @param M slope of critical state line
//  * @param lam slope of virgin consolidation line
//  * @param kap slope of swelling line
//  * @param Pt tensile yield hydrostatic stress
//  * @param beta parameter related to size of outer diameter of ellipse
//  * @param Vs volume of the solid
//  * @param mat_id material id
//  * @param do_update_history  variable if history should be updated
//  * @param is_velgrad_strain_increment variable if the velocity gradient is
//  * to be used as a strain increment
//  * @param tid thread id (particle id)
//  */
__device__ __host__ inline void update_dp_rheo(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
    const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
    Real *particles_alpha_gpu, Real *pc_gpu,
    Real *solid_volume_fraction_gpu,
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

 
  // Total volumetric strain
  const Real deps_v = deps_curr.trace();

  const Real solid_volume_fraction = Vs/particles_volume_gpu[tid] ;

  const Real solid_volume_fraction_original = Vs/particles_volume_original_gpu[tid];
  
  solid_volume_fraction_gpu[tid] = solid_volume_fraction;
          

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


  /// start DP cap
  const Real eta = sqrt(1. / 3.) * M;
  const Real Phi_cone_trail = compute_yield_function_cone(p_trail, Pt, J2_trail, eta);

  // check if yield functions are within elastic domain

  if (Phi_cone_trail <= (Real)0.0) {
    particles_stresses_gpu[tid] = s_trail + p_trail * Matrix3r::Identity();

    if (do_update_history) {
      particles_eps_e_gpu[tid] = eps_e_trail;
    }
    return;
  }

  // 4. Newton-Raphson iteration
  double p_next = p_trail;
  Matrix3r s_next = s_trail;

  double dgamma_a_next = 0.0;

  return_mapping_cone(p_next, s_next, dgamma_a_next, s_trail,
                      J2_trail, p_trail, eta_overline, eta,
                      bulk_modulus, shear_modulus, Phi_cone_trail);

  const bool is_apex =
      sqrt(J2_trail) - shear_modulus * dgamma_a_next < (Real)0.0;

  // check if we project bact to apex
  if (is_apex) {

    return_mapping_apex(p_next, s_next, dgamma_a_next, p_trail,
                        eta_overline, eta, Pt, bulk_modulus);
  }

  update_plasticity(s_next, p_next, s_ref, p_ref,
                        shear_modulus, bulk_modulus, do_update_history,
                        particles_stresses_gpu, particles_eps_e_gpu,
                        tid);
  // Real a1 = 0.021;
  // Real a2 = 0.095;
  // // Real a2 = 10.0;
  // Real a4 = 0.2;
  // Real dotgamma_0 = 0.1;
  // Real phi_c = 0.6;
  // Real d =1.0;
  // Real k = 1.0;
  // Real p_inert_guess = 0.0;


  // double dilatancy_function = 999999999999999.0;

  // do {
  // Real p_star = -p_next + p_inert_guess;
  // // printf("p_star %f p_inert_guess %f p_next %f \n", p_star, p_inert_guess, p_next);
  // // // Real p_star_a = p_star/a1; 
  // // double A =p_star/a1;
  // // double B = 3.0/2.0;

  // // printf("A %f B %f \n", A, B);
  // dilatancy_function = pow(p_star/a1,3.0/2.0) -  dotgamma_0*pow(a2/p_star,1.0/2.0) -solid_volume_fraction + phi_c;
  // // printf("dilatancy_function %f \n", dilatancy_function);
  // double residue = (d/k)*((3/2)*(1/a2)*pow(p_star/a2,1.0/2.0) +  (1/2)*pow(a1,1.0/2.0)*pow(p_star,-3.0/2.0)*dotgamma_0 );

  // p_inert_guess = p_inert_guess - dilatancy_function/residue;

  // printf("p_inert_guess %f \n", p_inert_guess);
  // } while(abs(dilatancy_function) > 1e-6);

  // // printf("p_inert_guess %f \n", p_inert_guess);

  // if(std::isnan(p_inert_guess)){
  //   // printf("p_inert_guess %f \n", p_inert_guess);
  //   return;
  // }

  // printf("p_inert_guess %f \n", p_inert_guess);
  // // particles_stresses_gpu[tid] = - p_inert_guess * Matrix3r::Identity();

  


}

#ifdef CUDA_ENABLED
__global__ void KERNEL_STRESS_UPDATE_DP_RHEO(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
    const Real *particles_volume_gpu, const Real *particles_volume_original_gpu,
    Real *particles_alpha_gpu, Real *pc_gpu,
    Real *solid_volume_fraction_gpu,
    const Matrixr *particles_velocity_gradient_gpu,
    const uint8_t *particle_colors_gpu, const Matrix3r *stress_ref_gpu,
    const Real bulk_modulus, const Real shear_modulus, const Real M,
    const Real lam, const Real kap, const Real Pt, const Real beta,
    const Real Vs, const Real eta_overline, const int mat_id,
    const bool do_update_history, const bool is_velgrad_strain_increment,
    const int num_particles) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads


update_dp_rheo(
    particles_stresses_gpu,
    particles_eps_e_gpu,
    particles_volume_gpu,
    particles_volume_original_gpu,
    particles_alpha_gpu,
    pc_gpu,
    solid_volume_fraction_gpu,
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


  // update_dp_rheo(particles_stresses_gpu, particles_eps_e_gpu,
  //                         particles_volume_gpu, particles_volume_original_gpu,
  //                         particles_alpha_gpu, pc_gpu,
  //                         particles_velocity_gradient_gpu, particle_colors_gpu,
  //                         stress_ref_gpu, bulk_modulus, shear_modulus, M, lam,
  //                         kap, Pt, beta, Vs, const Real eta_overline, mat_id,
  //                         do_update_history, is_velgrad_strain_increment, tid);

  // __device__ __host__ inline void update_modifiedcamclay(
  //     Matrix3r * particles_stresses_gpu, Matrixr * particles_eps_e_gpu,
  //     const Real *particles_volume_gpu,
  //     const Real *particles_volume_original_gpu, Real *particles_alpha_gpu,
  //     Real *pc_gpu, const Matrixr *particles_velocity_gradient_gpu,
  //     const uint8_t *particle_colors_gpu, const Matrix3r *stress_ref_gpu,
  //     const Real bulk_modulus, const Real shear_modulus, const Real M,
  //     const Real lam, const Real kap,  const Real Pt,
  //     const Real beta, const Real Vs, const int mat_id,
  //     const bool do_update_history, const bool is_velgrad_strain_increment,
  //     const int tid)

  //     update_linearelastic(particles_stresses_gpu, particles_F_gpu,
  //                          particles_colors_gpu, particles_is_active_gpu,
  //                          shear_modulus, bulk_modulus, mat_id, tid);
}
#endif

} // namespace pyroclastmpm