/*
[1] https://en.wikipedia.org/wiki/Infinitesimal_strain_theory
[2] de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen. Computational
methods for plasticity: theory and applications. John Wiley & Sons, 2011.
*/
// #include <Eigen/Eigenvalues>

__device__ __host__ inline void
update_vonmises(Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
                Real *particles_acc_eps_p_gpu,
                const Matrixr *particles_velocity_gradient_gpu,
                const Matrixr *particles_F_gpu,
                const uint8_t *particle_colors_gpu, const Real bulk_modulus,
                const Real shear_modulus, const Real yield_stress, const Real H,
                const int mat_id, const int tid) {

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
  Matrixr F = particles_F_gpu[tid]; // deformation gradient

  // (total strain current step) infinitesimal strain assumptions [1]
  const Matrixr deps_curr =
      0.5 * (vel_grad + vel_grad.transpose()) * dt; // pseudo strain rate
  const Matrixr eps_curr = 0.5 * (F.transpose() + F) - Matrixr::Identity();

  // elastic strain (previous step)
  const Matrixr eps_e_prev = particles_eps_e_gpu[tid];

  // trail step, eq (7.21) [2]
  const Matrixr eps_e_n_tr = eps_e_prev + deps_curr; // trial elastic strain
  const Real acc_eps_p_tr =
      particles_acc_eps_p_gpu[tid]; // accumulated plastic strain (previous
                                    // step)

  // hydrostatic stress eq (7.82) and volumetric strain eq (3.90) [2]
  const Real eps_e_v_trail = eps_e_prev.trace();
  const Real p_trail = bulk_modulus * eps_e_v_trail;

  // deviatoric stress (7.82) and strain eq (3.114) [2]
  const Matrixr eps_e_dev_trail =
      eps_e_n_tr - (1 / 3.) * eps_e_v_trail * Matrixr::Identity();
  const Matrixr s_trail = 2. * shear_modulus * eps_e_dev_trail;

  // isotropic linear hardening eq (6.170) [2]
  const Real sigma_y_trail = yield_stress + H * acc_eps_p_tr;

  // yield function eq (6.106) and (6.110) [2]
  const Real q_trail = sqrt(3 * 0.5 * (s_trail * s_trail.transpose()).trace());
  const Real Phi = q_trail - sigma_y_trail;

  // if stress is in feasible region elastic step eq (7.84)
  if (Phi <= 0) {
    particles_stresses_gpu[tid] = s_trail + p_trail * Matrixr::Identity();
    particles_eps_e_gpu[tid] = eps_e_n_tr;
    particles_acc_eps_p_gpu[tid] = acc_eps_p_tr;
    return;
  }

  // otherwise do return mapping - box 7.4 [2]
  // find plastic multiplier dgamma, such that yield function is approximately
  // zero using newton raphson method
  double dgamma = 0.0;

  double psi_approx, acc_eps;

  const double tol = 1e-7;

  int iter = 0; // debug purposes
  do {
    // simplified implicit return mapping equation (7.91)
    // uses the fact that VM flow vector is purely deviatoric
    // (pressure-independent)
    acc_eps = acc_eps_p_tr + dgamma;

    // isotropic linear hardening eq (6.170) [2]
    const double sigma_y = yield_stress + H * acc_eps;
    psi_approx = q_trail - 3.0 * shear_modulus * dgamma - sigma_y;

    const double d = -3.0 * shear_modulus - H; // residual of yield function

    dgamma = dgamma - psi_approx / d;

    // printf("dgamma: %f psi_approx %f iter %d H %f \n", dgamma,
    // psi_approx,iter);
    iter += 1;
  } while (psi_approx > tol);

  const Real p_curr =
      p_trail; // since von mises yield function is an isotropic function

  const Matrixr s_curr = (1 - 3.0 * shear_modulus * dgamma / q_trail) * s_trail;

  const Matrixr sigma_curr = s_curr + p_curr * Matrixr::Identity();
  const Matrixr eps_e_curr = s_curr / (2.0 * shear_modulus) +
                             (1. / 3.) * eps_e_v_trail * Matrixr::Identity();

  particles_stresses_gpu[tid] = sigma_curr;
  particles_eps_e_gpu[tid] = eps_e_curr;
  particles_acc_eps_p_gpu[tid] = acc_eps;
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_STRESS_UPDATE_VONMISES(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_eps_e_gpu,
    Real *particles_acc_eps_p_gpu,
    const Matrixr *particles_velocity_gradient_gpu,
    const Matrixr *particles_F_gpu, const uint8_t *particle_colors_gpu,
    const Real bulk_modulus, const Real shear_modulus, const Real yield_stress,
    const Real H, const int mat_id, const int num_particles) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_particles) {

    return;
  }

  update_vonmises(particles_stresses_gpu, particles_eps_e_gpu,
                  particles_acc_eps_p_gpu, particles_velocity_gradient_gpu,
                  particles_F_gpu, particle_colors_gpu, bulk_modulus,
                  shear_modulus, yield_stress, H, mat_id, tid);
}
#endif