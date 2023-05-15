#include "pyroclastmpm/materials/druckerprager/druckerpragermat.cuh"

#include <Eigen/Core>
#include <Eigen/SVD>

namespace pyroclastmpm
{

  /**
   * @brief global step counter
   *
   */
  extern int global_step_cpu;

  /**
   * @brief Construct a new Drucker Prager:: Drucker Prager object
   *
   * @param _density material density
   * @param _E Young's modulus
   * @param _pois Poisson's ratio
   * @param _friction_angle
   * @param _cohesion Hardening parameter
   * @param _vcs volume correction scalar
   */
  DruckerPrager::DruckerPrager(const Real _density,
                               const Real _E,
                               const Real _pois,
                               const Real _friction_angle,
                               const Real _cohesion,
                               const Real _vcs)
  {
    E = _E;
    pois = _pois;
    density = _density;

    shear_modulus = (1. / 2.) * E / (1. + pois);
    lame_modulus = (pois * E) / ((1. + pois) * (1. - 2. * pois));

    friction_angle = _friction_angle;
    cohesion = _cohesion;
    vcs = _vcs;

    name = "DruckerPrager";

    Real sin_phi = std::sin(friction_angle / 180. * 3.141592653);
    alpha = std::sqrt(2. / 3.) * 2. * sin_phi / (3. - sin_phi);
  }

  void DruckerPrager::stress_update(ParticlesContainer &particles_ref,
                                    int mat_id)
  {

    KERNEL_TRAIL_FE_DRUCKERPRAGER<<<particles_ref.launch_config.tpb,
                                    particles_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles_ref.F_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
        particles_ref.num_particles,
        mat_id);

    gpuErrchk(cudaDeviceSynchronize());

    // SVD cant be called inside kernels

    cpu_array<Matrixr> F_cpu = particles_ref.F_gpu;
    cpu_array<Matrixr> U_cpu;
    cpu_array<Matrixr> V_cpu;
    cpu_array<Vectorr> S_cpu;

    U_cpu.resize(particles_ref.num_particles);
    V_cpu.resize(particles_ref.num_particles);
    S_cpu.resize(particles_ref.num_particles);

    // #pragma omp parallel for num_threads(4)
    for (int pi = 0; pi < particles_ref.num_particles; pi++)
    {
      // if (particles_ref.colors_cpu[pi] == mat_id) {
      Eigen::JacobiSVD<Matrixr> svd(F_cpu[pi], Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV);
      U_cpu[pi] = svd.matrixU();
      V_cpu[pi] = svd.matrixV();
      S_cpu[pi] = svd.singularValues();
      // }
    }

    gpu_array<Matrixr> U_gpu = U_cpu;
    gpu_array<Matrixr> V_gpu = V_cpu;
    gpu_array<Vectorr> S_gpu = S_cpu;

    KERNEL_STRESS_UPDATE_DRUCKERPRAGER<<<particles_ref.launch_config.tpb,
                                         particles_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.F_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.logJp_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.pressures_gpu.data()),
        thrust::raw_pointer_cast(U_gpu.data()),
        thrust::raw_pointer_cast(V_gpu.data()),
        thrust::raw_pointer_cast(S_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
        alpha, // todo
        shear_modulus,
        lame_modulus,
        cohesion,
        vcs,
        particles_ref.num_particles,
        mat_id);

    gpuErrchk(cudaDeviceSynchronize());

    // exit(0);
  }
  /**
   * @brief outbound stress update for CPU
   *
   * @param stress stress tensor
   * @param Fe elastic deformation gradient
   * @param logJp log of the determinant of the plastic deformation gradient (volume correction)
   * @param Fp_tr trail plastic deformation gradeint (identity + dt * velgrad)
   * @param alpha Hardening parameter
   * @param dim dimension
   */
  void DruckerPrager::outbound_stress_update(Matrix3r &stress,
                                             Matrix3r &Fe,
                                             Real &logJp,
                                             const Matrix3r Fp_tr,
                                             const Real alpha,
                                             const int dim)
  {
    printf("outbound_stress_update called \n");

    // note let Fp_tr = (Matrix3r::Identity() + dt * velgrad)
    Matrix3r Fe_tr = Fp_tr * Fe;
    // printf("Fe_tr = %.16f %.16f %.16f \n", Fe_tr(0, 0), Fe_tr(1, 1), Fe_tr(2, 2));
    Eigen::JacobiSVD<Matrix3r> svd(Fe_tr, Eigen::ComputeFullU |
                                              Eigen::ComputeFullV);
    const Matrix3r U = svd.matrixU();
    const Matrix3r V = svd.matrixV();
    const Vector3r S = svd.singularValues();
    // printf("S = %.16f %.16f %.16f \n", S(0, 0), S(1, 1), S(2, 2));
    Vector3r eps;

    eps(0) = log(max(abs(S(0)), 1.e-4));
    eps(1) = log(max(abs(S(1)), 1.e-4));
    eps(2) = log(max(abs(S(2)), 1.e-4));

    // printf("eps = %.16f %.16f %.16f \n", eps(0, 0), eps(1, 1), eps(2, 2));

    Real sum_eps = eps.sum() + logJp;
    Vector3r eps_hat = eps - Vector3r::Ones() * (sum_eps / (Real)(dim));

    // printf("eps_hat = %.16f %.16f %.16f \n", eps_hat(0, 0), eps_hat(1, 1), eps_hat(2, 2));

    Real eps_norm = eps.norm();
    Real eps_hat_norm = eps_hat.norm();
    // printf("eps_hat_norm %.16f \n", eps_hat_norm);

    Real dlogJp = 0.;
    Vector3r S_new;

    if (sum_eps > 0.)
    {
      // Project to start of cone. Tension

      S_new(0) = exp(cohesion);
      S_new(1) = exp(cohesion);
      S_new(2) = exp(cohesion);
      dlogJp = vcs * sum_eps;
    }
    else
    {
      Real delta_gamma =
          eps_hat_norm + alpha * sum_eps *
                             (dim * lame_modulus + 2. * shear_modulus) /
                             (2 * shear_modulus);
      printf("delta_gamma: %.16f \n", delta_gamma);
      printf("alpha %.16f \n", alpha);

      if (delta_gamma <= 0)
      {
        // Elastic deformation gradient already on the yield surface
        Vector3r H = eps + Vector3r::Ones() * cohesion;

        S_new(0) = exp(H(0));
        S_new(1) = exp(H(1));
        S_new(2) = exp(H(2));

        // printf("case 1 \n");
        // printf("H = %.16f %.16f %.16f \n", H(0, 0), H(1, 1), H(2, 2));
        // printf("S_new = %f %f %f \n", S_new(0, 0), S_new(1, 1), S_new(2, 2));
      }
      else
      {
        printf("case 3 \n");
        Vector3r H = eps - delta_gamma * (eps_hat / eps_hat_norm) + Vector3r::Ones() * cohesion;

        // printf("H = %.16f %.16f %.16f \n", H(0, 0), H(1, 1), H(2, 2));

        S_new(0) = exp(H(0));
        S_new(1) = exp(H(1));
        S_new(2) = exp(H(2));

        // printf("S_new = %f %f %f \n", S_new(0, 0), S_new(1, 1), S_new(2, 2));
      }
    }

    logJp += dlogJp;

    const Matrix3r S_diag = S.asDiagonal();

    const Matrix3r V_T = V.transpose();

    // Reproject to find new Fe and Fp
    Fe = U * S_diag * V_T;

    // printf("Fe = [[%.16f, %.16f, %.16f], [%.16f, %.16f, %.16f], [%.16f, %.16f, %.16f]] \n", Fe(0, 0), Fe(0, 1), Fe(0, 2), Fe(1, 0), Fe(1, 1), Fe(1, 2), Fe(2, 0), Fe(2, 1), Fe(2, 2));

    const Matrix3r S_inverse = S_diag.inverse();

    Matrix3r Se_log = Matrix3r::Zero();
    Se_log(0, 0) = log(S_diag(0, 0));
    Se_log(1, 1) = log(S_diag(1, 1));
    Se_log(2, 2) = log(S_diag(2, 2));

    Matrix3r Se_update = 2.0 * shear_modulus * S_inverse * Se_log + lame_modulus * Se_log.trace() * S_inverse;

    Matrix3r T = U * Se_update * V_T;

    printf("T = [[%.16f, %.16f, %.16f], [%.16f, %.16f, %.16f], [%.16f, %.16f, %.16f]] \n", T(0, 0), T(0, 1), T(0, 2), T(1, 0), T(1, 1), T(1, 2), T(2, 0), T(2, 1), T(2, 2));

    stress = -T / Fe.determinant(); // Cauchy stress

    // printf("stress = [[%.16f, %.16f, %.16f], [%.16f, %.16f, %.16f], [%.16f, %.16f, %.16f]] \n", stress(0, 0), stress(0, 1), stress(0, 2), stress(1, 0), stress(1, 1), stress(1, 2), stress(2, 0), stress(2, 1), stress(2, 2));
  }

  DruckerPrager::~DruckerPrager() {}

} // namespace pyroclastmpm