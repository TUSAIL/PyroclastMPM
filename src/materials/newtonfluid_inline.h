__device__ __host__ inline void stress_update_newtonfluid(
    Matrix3r *particles_stresses_gpu,
    const Matrixr *particles_velocity_gradients_gpu,
    const Real *particles_masses_gpu, const Real *particles_volumes_gpu,
    const Real *particles_volumes_original_gpu,
    const uint8_t *particles_colors_gpu, const bool *particles_is_active_gpu,
    const Real viscosity, const Real bulk_modulus, const Real gamma,
    const int mat_id, const int tid) {

  if (!particles_is_active_gpu[tid]) {
    return;
  }

  const int particle_color = particles_colors_gpu[tid];

  if (particle_color != mat_id) {
    return;
  }

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

  // printf("runs after check");
  const Matrixr vel_grad = particles_velocity_gradients_gpu[tid];
  const Matrixr vel_grad_T = vel_grad.transpose();
  const Matrixr strain_rate = 0.5 * (vel_grad + vel_grad_T) * dt_gpu;

  Matrixr deviatoric_part =
      strain_rate - (1. / 3) * strain_rate.trace() * Matrixr::Identity();

  // printf("deviatoric_part: %f\n", deviatoric_part);
  const Real density = particles_masses_gpu[tid] / particles_volumes_gpu[tid];

  // printf("masses %f\n", particles_masses_gpu[tid]);
  // printf("volumes %f\n", particles_volumes_gpu[tid]);

  // printf("density: %f\n", density);
  const Real density_original =
      particles_masses_gpu[tid] / particles_volumes_original_gpu[tid];
  Real mu = density / density_original;

  Real pressure = bulk_modulus * (pow(mu, gamma) - 1.);

  Matrixr cauchy_stress =
      2 * viscosity * deviatoric_part - pressure * Matrixr::Identity();

#if DIM == 3
  particles_stresses_gpu[tid] = cauchy_stress;
#else
  particles_stresses_gpu[tid].block(0, 0, DIM, DIM) = cauchy_stress;
#endif
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