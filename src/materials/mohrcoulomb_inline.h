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

Currently two problems

(1) return mapping side, if tension is applied, edge is wrong size
 if compression is applied we switch.. maybe check direction of
 volumetric strain to determine which edge to use

 (2) at low diltancy angles we have weird effects
 maybe sfa and sda is switched?
also return mapping on edge

*/

using Matrix2hp = Eigen::Matrix<double, 2, 2>;
using Vector2hp = Eigen::Matrix<double, 2, 1>;
using Vector3hp = Eigen::Matrix<double, 3, 1>;

__device__ __host__ inline bool
return_mapping_mainplane(Matrix3r &stresses, Matrixr &eps_e, Real &acc_eps_p,
                         const Matrix3r principal_dir, const Real Phi_tr,
                         const Real acc_eps_p_tr, const Vector3r pstress_tr,
                         const Real K, const Real G, const Real sda,
                         const Real sfa, const Real cda, const Real cfa,
                         const Real a, const Real H, const Real cohesion) {

  double dgamma = 0.0;

  double Phi_plane = Phi_tr;

  // Do return mapping see BOX 8.5 [2]
  double acc_eps_p_next;
  // printf("before return map main \n");
  do {

    // residual derivative of yield function on main plane
    const double d = -a - 4 * H * cfa * cfa;

    dgamma = dgamma - Phi_plane / d;

    // cohesion approximation related to eq (8.70) [2]
    acc_eps_p_next = acc_eps_p_tr + 2 * cfa * dgamma;

    // linear strain hardening
    const double cohesion_approx = cohesion + H * acc_eps_p_next;

    // eq (8.71) [2]
    Phi_plane = pstress_tr(0) - pstress_tr(2) +
                (pstress_tr(0) + pstress_tr(2)) * sfa -
                2. * cohesion_approx * cfa - a * dgamma;

  } while (abs(Phi_plane) > 1.e-6);

  // updated principal stresses eq (8.69) [2]
  Vector3r pstress_curr = Vector3r::Zero();

  pstress_curr(0) =
      pstress_tr(0) - dgamma * (2. * G * (1. + (1. / 3.) * sda) + 2. * K * sda);
  pstress_curr(1) = pstress_tr(1) + dgamma * ((4. / 3.) * G - 2. * K) * sda;
  pstress_curr(2) =
      pstress_tr(2) + dgamma * (2. * G * (1. - (1. / 3.) * sda) - 2. * K * sda);

  if ((trunc(pstress_curr(0)) >= trunc(pstress_curr(1))) &&
      (trunc(pstress_curr(1)) >= trunc(pstress_curr(2)))) {

    // update stress, elastic strain, accumulated plastic strain
    stresses =
        principal_dir * pstress_curr.asDiagonal() * principal_dir.inverse();

    // updated pressure
    Real p_next = stresses.trace() / 3.0;

    // updated deviatoric stress
    Matrix3r s_next = stresses - p_next * Matrix3r::Identity();

    // updated elastic strain
    eps_e = s_next.block(0, 0, DIM, DIM) * 1. / (2. * G) +
            p_next * 1. / (3. * K) * Matrixr::Identity();

    // hardening parameters
    acc_eps_p = acc_eps_p_next;

    return true;
  }
  return false;
}

__device__ __host__ inline bool
return_mapping_edge(Matrix3r &stresses, Matrixr &eps_e, Real &acc_eps_p,
                    const Matrix3r principal_dir, const Real cohesion_tr,
                    const Real acc_eps_p_tr, const Vector3r pstress_tr,
                    const Real K, const Real G, const Real sda, const Real sfa,
                    const Real cda, const Real cfa, const Real a, const Real H,
                    const Real cohesion, const bool is_right_edge)

{

  // Sub-gradient gives two possible directions
  // therefore, we need to check both directions
  Vector2hp dgamma = Vector2hp::Zero();

  Vector2hp Phi_edge;

  const Real sigma_1 =
      pstress_tr(0) - pstress_tr(2) + (pstress_tr(0) + pstress_tr(2)) * sfa;

  Real sigma_2;

  if (is_right_edge) {
    sigma_2 =
        pstress_tr(0) - pstress_tr(1) + (pstress_tr(0) + pstress_tr(1)) * sfa;
  } else {
    sigma_2 =
        pstress_tr(1) - pstress_tr(2) + (pstress_tr(1) + pstress_tr(2)) * sfa;
  }

  Phi_edge(0) = sigma_1 - 2. * cohesion_tr * cfa;
  Phi_edge(1) = sigma_2 - 2. * cohesion_tr * cfa;

  double acc_eps_p_next = acc_eps_p_tr;

  // constants
  double b;
  // printf("before return map edge \n");
  if (is_right_edge) {
    // eq (8.77) [2]
    b = 2 * G * (1. + sfa + sda - (1. / 3.) * sfa * sda) + 4. * K * sfa * sda;
  } else {
    // eq (8.80) [2]
    b = 2 * G * (1. - sfa - sda - (1 / 3.) * sfa * sda) + 4. * K * sfa * sda;
  }

  do {

    // residual of yield function Bo 8.6 [2]

    Matrix2hp d;

    d(0, 0) = -a - 4. * H * cfa * cfa;
    d(1, 1) = d(0, 0);

    d(0, 1) = -b - 4 * H * cfa * cfa;
    d(1, 0) = d(0, 1);

    dgamma = dgamma - d.inverse() * Phi_edge;

    //   // eq (8.75)
    acc_eps_p_next = acc_eps_p_tr + 2 * cfa * (dgamma(0) + dgamma(1));

    const double cohesion_approx = cohesion + H * acc_eps_p_next;

    // yield surface eq (8.76) [2]
    Phi_edge(0) =
        sigma_1 - 2. * cohesion_approx * cfa - a * dgamma(0) - b * dgamma(1);
    Phi_edge(1) =
        sigma_2 - 2. * cohesion_approx * cfa - a * dgamma(1) - b * dgamma(0);

  } while (abs(Phi_edge(0)) + abs(Phi_edge(1)) > 1.e-7);

  Vector3r pstress_curr = Vector3r::Zero();

  if (is_right_edge) {

    pstress_curr(0) =
        pstress_tr(0) - (2. * G * (1. + (1. / 3.) * sda) + 2. * K * sda) *
                            (dgamma(0) + dgamma(1));
    pstress_curr(1) =
        pstress_tr(1) + ((4. / 3.) * G - 2. * K) * sda * dgamma(0) +
        (2. * G * (1. - (1. / 3.) * sda) - 2. * K * sda) * dgamma(1);

    pstress_curr(2) =
        pstress_tr(2) +
        (2. * G * (1. - (1. / 3.) * sda) - 2. * K * sda) * dgamma(0) +
        ((4. / 3.) * G - 2. * K) * sda * dgamma(1);

  } else {

    // note there is a mistake in the book
    // should be dgamma^a and dgamma^b
    pstress_curr(0) =
        pstress_tr(0) -
        (2. * G * (1. + (1. / 3.) * sda) + 2. * K * sda) * dgamma(0) +
        ((4. / 3.) * G - 2. * K) * sda * dgamma(1);

    // checked
    pstress_curr(1) =
        pstress_tr(1) + ((4. / 3.) * G - 2. * K) * sda * dgamma(0) -
        (2. * G * (1. + (1. / 3.) * sda) + 2. * K * sda) * dgamma(1);

    // checked
    pstress_curr(2) =
        pstress_tr(2) + (2. * G * (1. - (1. / 3.) * sda) - 2. * K * sda) *
                            (dgamma(0) + dgamma(1));
  }

  // // check validity and return
  if ((trunc(pstress_curr(0)) >= trunc(pstress_curr(1))) &&
      (trunc(pstress_curr(1)) >= trunc(pstress_curr(2)))) {

    stresses =
        principal_dir * pstress_curr.asDiagonal() * principal_dir.transpose();

    // updated pressure
    Real p_next = stresses.trace() / 3.0;

    // updated deviatoric stress
    Matrix3r s_next = stresses - p_next * Matrix3r::Identity();

    // updated elastic strain
    eps_e = s_next.block(0, 0, DIM, DIM) * 1. / (2. * G) +
            p_next * 1. / (3. * K) * Matrixr::Identity();

    // hardening parameters
    acc_eps_p = acc_eps_p_next;

    return true;
  }
  printf(
      "do not pass validity check pstress_curr %f %f %f is_right_edge %d \n ",
      pstress_curr(0), pstress_curr(1), pstress_curr(2), is_right_edge);
  // exit(1);
  return false;
}

__device__ __host__ inline void
return_mapping_apex(Matrix3r &stresses, Matrixr &eps_e, Real &acc_eps_p,
                    const Real Phi_tr, const Real acc_eps_p_tr, const Real p_tr,
                    const Real K, const Real G, const Real sda, const Real sfa,
                    const Real cda, const Real cfa, const Real H,
                    const Real cohesion) {
  // Update return mapping along the hydrostatic axis

  // solve for volumetric plastic strain increment
  double deps_p_v = 0;

  double Phi_apex = Phi_tr;

  double p_next;

  double acc_eps_p_next;

  const double alpha = cfa / sda;

  const double cotf = cfa / sfa; // // cot (fric_angle)

  do {
    // eq (8.84) [2]
    acc_eps_p_next = acc_eps_p_tr + alpha * deps_p_v;

    const double cohesion_approx = cohesion + H * acc_eps_p_next;

    p_next = p_tr - K * deps_p_v;

    Phi_apex = cohesion_approx * cotf - p_next;

    const double d = (H * cfa * cotf) / sda + K;

    deps_p_v = deps_p_v - Phi_apex / d;
    printf("Phi_apex %f\n", Phi_apex);

  } while (abs(Phi_apex) > 1.e-6);

  // update stress, elastic strain, accumulated plastic strain

  stresses = p_next * Matrix3r::Identity();
  // updated elastic strain
  eps_e = p_next * 1. / (3. * K) * Matrixr::Identity();

  // hardening parameters
  acc_eps_p = acc_eps_p_next;
}

__device__ __host__ inline void update_mohrcoulomb(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
    Real *particles_acc_eps_p_gpu,
    const Matrixr *particles_velocity_gradient_gpu,
    const Matrixr *particles_F_gpu, const uint8_t *particle_colors_gpu,
    const Real K, const Real G, const Real cohesion, const Real fric_angle,
    const Real dil_angle, const Real H, const int mat_id, const int tid) {

  const int particle_color = particle_colors_gpu[tid];

  if (particle_color != mat_id) {
    return;
  }

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

  const Matrixr vel_grad = particles_velocity_gradient_gpu[tid];

  // Matrixr F = particles_F_gpu[tid]; // deformation gradient
  // const Matrixr eps_curr = 0.5 * (F.transpose() + F) - Matrixr::Identity();

  // (total strain current step) infinitesimal strain assumptions [1]
  const Matrixr deps_curr =
      0.5 * (vel_grad + vel_grad.transpose()) * dt; // pseudo strain rate

  // elastic strain (previous step)
  const Matrixr eps_e_prev = particles_eps_e_gpu[tid];

  // trail step, eq (7.21) [2]
  const Matrixr eps_e_tr = eps_e_prev + deps_curr; // trial elastic strain

  const Real acc_eps_p_tr =
      particles_acc_eps_p_gpu[tid]; // accumulated plastic strain (previous
                                    // step)

  // volumetric strain eq (3.90) and hydrostatic stress eq (7.82) [2]
  const Real eps_e_vol_tr = eps_e_tr.trace();

  const Real p_tr = K * eps_e_vol_tr;

  // deviatoric strain eq (3.114) and stress (7.82) [2]
  const Matrixr eps_e_dev_tr =
      eps_e_tr - (1 / 3.) * eps_e_vol_tr * Matrixr::Identity();

  const Matrixr s_tr = 2. * G * eps_e_dev_tr;

  // isotropic linear hardening eq (6.170) [2]
  // TODO call it initial cohesion
  const Real cohesion_tr = cohesion + H * acc_eps_p_tr;

  Matrix3r sigma_tr = Matrix3r::Zero();
  sigma_tr.block(0, 0, DIM, DIM) = s_tr;
  sigma_tr += p_tr * Matrix3r::Identity();

  // Get eigen values using spectral decomposition
  Eigen::SelfAdjointEigenSolver<Matrix3r> spectral_decomp;
  spectral_decomp.computeDirect(sigma_tr);

  // trail principal stresses
  Vector3r pstress_tr = spectral_decomp.eigenvalues();

  // SelfAdjoint sorts from smallest
  // to largest, we want largest to
  // smallest
  pstress_tr = pstress_tr.reverse().eval();

  const Real sfa = sin(fric_angle);
  const Real cfa = cos(fric_angle);

  // yield function
  Real Phi_tr = pstress_tr(0) - pstress_tr(2) +
                (pstress_tr(0) + pstress_tr(2)) * sfa - 2. * cohesion_tr * cfa;

  // if stress is in feasible region elastic step eq (7.84)
  if (Phi_tr <= 0) {
    particles_stresses_gpu[tid] = sigma_tr;
    particles_eps_e_gpu[tid] = eps_e_tr;
    particles_acc_eps_p_gpu[tid] = acc_eps_p_tr;
    return;
  }

  Matrix3r principal_dir = spectral_decomp.eigenvectors();

  principal_dir = principal_dir.rowwise().reverse().eval();

  const Real sda = sin(dil_angle);

  const Real cda = cos(dil_angle);

  // constant used in return mapping main plane or edges eq (8.72) [2]
  Real a = 4. * G * (1. + (1. / 3.) * sfa * sda) + 4 * K * sfa * sda;

  // 1. Check if stress lies on main plane & update
  const bool is_mainplane = return_mapping_mainplane(
      particles_stresses_gpu[tid], particles_eps_e_gpu[tid],
      particles_acc_eps_p_gpu[tid], principal_dir, Phi_tr, acc_eps_p_tr,
      pstress_tr, K, G, sda, sfa, cda, cfa, a, H, cohesion);

  if (is_mainplane) {
    return;
  }
  // 2. Check if stress lies on either left or right edge & update

  // // check if stress lies on right edge
  // const Real right_extent = (1. - sda) * pstress_tr(0) +
  //                           (1. + sda) * pstress_tr(1) - 2. * pstress_tr(2);

  // bool is_right_edge = right_extent > 0;

  const bool is_right_edge = return_mapping_edge(
      particles_stresses_gpu[tid], particles_eps_e_gpu[tid],
      particles_acc_eps_p_gpu[tid], principal_dir, cohesion_tr, acc_eps_p_tr,
      pstress_tr, K, G, sda, sfa, cda, cfa, a, H, cohesion, true);

  if (is_right_edge) {
    return;
  }

  const bool is_left_edge = return_mapping_edge(
      particles_stresses_gpu[tid], particles_eps_e_gpu[tid],
      particles_acc_eps_p_gpu[tid], principal_dir, cohesion_tr, acc_eps_p_tr,
      pstress_tr, K, G, sda, sfa, cda, cfa, a, H, cohesion, false);

  if (is_left_edge) {
    return;
  }
  return_mapping_apex(particles_stresses_gpu[tid], particles_eps_e_gpu[tid],
                      particles_acc_eps_p_gpu[tid], Phi_tr, acc_eps_p_tr, p_tr,
                      K, G, sda, sfa, cda, cfa, H, cohesion);
}