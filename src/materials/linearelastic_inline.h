__device__ __host__ inline void update_linearelastic(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_velocity_gradient_gpu,
    const Real *particles_volumes_gpu, const Real *particles_masses_gpu,
    const uint8_t *particles_colors_gpu, bool *particles_is_active_gpu,
    const Real shear_modulus, const Real lame_modulus, const int mat_id,
    const int tid) {

  const int particle_color = particles_colors_gpu[tid];
  if (!particles_is_active_gpu[tid]) {
    return;
  }

  if (particle_color != mat_id) {
    return;
  }

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

  const Matrixr vel_grad = particles_velocity_gradient_gpu[tid];
  const Matrixr velgrad_T = vel_grad.transpose();
  const Matrixr deformation_matrix =
      0.5 * (vel_grad + velgrad_T); // infinitesimal strain
  const Matrixr strain_increments =
      deformation_matrix * dt; // pseudeo strain rate

#if DIM == 3
  Matrixr cauchy_stress = particles_stresses_gpu[tid];
#else
  Matrix3r cauchy_stress_3d = particles_stresses_gpu[tid];
  Matrixr cauchy_stress = cauchy_stress_3d.block(0, 0, DIM, DIM);
#endif

  cauchy_stress += lame_modulus * strain_increments * Matrixr::Identity() +
                   2. * shear_modulus * strain_increments;
#if DIM == 3
  particles_stresses_gpu[tid] = cauchy_stress;
#else
  cauchy_stress_3d.block(0, 0, DIM, DIM) = cauchy_stress;
  particles_stresses_gpu[tid] = cauchy_stress_3d;
#endif
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_STRESS_UPDATE_LINEARELASTIC(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_velocity_gradient_gpu,
    const Real *particles_volumes_gpu, const Real *particles_masses_gpu,
    const uint8_t *particles_colors_gpu, bool *particles_is_active_gpu,
    const int num_particles, const Real shear_modulus, const Real lame_modulus,
    const int mat_id) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  update_linearelastic(particles_stresses_gpu, particles_velocity_gradient_gpu,
                       particles_volumes_gpu, particles_masses_gpu,
                       particles_colors_gpu, particles_is_active_gpu,
                       shear_modulus, lame_modulus, mat_id, tid);
}
#endif