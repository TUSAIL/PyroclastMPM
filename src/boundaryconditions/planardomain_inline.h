

__host__ __device__ inline void apply_planardomain(
    Vectorr *particles_forces_external_gpu,
    const Vectorr *particles_positions_gpu,
    const Vectorr *particles_velocities_gpu, const Real *particles_volumes_gpu,
    const Real *particle_masses_gpu, const Vectorr axis0_friction,
    const Vectorr axis1_friction, const Vectorr domain_start,
    const Vectorr domain_end, const int mem_index) {

  const Vectorr pos = particles_positions_gpu[mem_index];
  const Vectorr vel = particles_velocities_gpu[mem_index];

  const Vectorr vel_norm = vel.normalized();

  const Real vol = particles_volumes_gpu[mem_index];

  const Real Radius = 0.5 * pow(vol, 1. / DIM);

  const Real mass = particle_masses_gpu[mem_index];

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

#if DIM == 3
  const Vectorr normals0[6] = {Vectorr({1, 0, 0}), Vectorr({0, 1, 0}),
                               Vectorr({0, 0, 1})};
  const Vectorr normals1[6] = {Vectorr({-1, 0, 0}), Vectorr({0, -1, 0}),
                               Vectorr({0, 0, -1})};

#elif DIM == 2
  const Vectorr normals0[2] = {Vectorr({1, 0}), Vectorr({0, 1})};
  const Vectorr normals1[2] = {Vectorr({-1, 0}), Vectorr({0, -1})};

#else
  const Vectorr normals0[1] = {Vectorr(1)};
  const Vectorr normals1[1] = {Vectorr(-1)};
#endif

  const Vectorr overlap0 = Vectorr::Ones() * Radius - (pos - domain_start);

#pragma unroll
  for (int i = 0; i < DIM; i++) {
    if (overlap0[i] > 0) {
      const Vectorr vel_depth =
          (overlap0[i] * normals0[i]).dot(vel) * normals0[i];
      const Vectorr fric_term =
          normals0[i] -
          axis0_friction(i) * (vel - normals0[i].dot(vel) * normals0[i]);
      particles_forces_external_gpu[mem_index] +=
          (mass / pow(dt, 2.)) * overlap0[i] * fric_term;
    }
  }
  const Vectorr overlap1 = Vectorr::Ones() * Radius - (domain_end - pos);

#pragma unroll
  for (int i = 0; i < DIM; i++) {
    if (overlap1[i] > 0) {
      const Vectorr vel_depth =
          (overlap1[i] * normals1[i]).dot(vel) * normals1[i];
      const Vectorr fric_term =
          normals1[i] -
          axis1_friction(i) * (vel - normals1[i].dot(vel) * normals1[i]);
      particles_forces_external_gpu[mem_index] +=
          (mass / pow(dt, 2.)) * overlap1[i] * fric_term;
    }
  }
}

#ifdef CUDA_ENABLED
__global__ void KERNELS_APPLY_PLANARDOMAIN(
    Vectorr *particles_forces_external_gpu,
    const Vectorr *particles_positions_gpu,
    const Vectorr *particles_velocities_gpu, const Real *particles_volumes_gpu,
    const Real *particle_masses_gpu, const Vectorr axis0_friction,
    const Vectorr axis1_friction, const Vectorr domain_start,
    const Vectorr domain_end, const int num_particles) {
  const int mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (mem_index >= num_particles) {
    return;
  }

  apply_planardomain(particles_forces_external_gpu, particles_positions_gpu,
                     particles_velocities_gpu, particles_volumes_gpu,
                     particle_masses_gpu, axis0_friction, axis1_friction,
                     domain_start, domain_end, mem_index);
}

#endif