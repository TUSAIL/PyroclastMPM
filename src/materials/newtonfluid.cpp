#include "pyroclastmpm/materials/newtonfluid.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern Real dt_cpu;
#endif

#include "newtonfluid_inline.h"

NewtonFluid::NewtonFluid(const Real _density, const Real _viscosity,
                         const Real _bulk_modulus, const Real _gamma) {
  viscosity = _viscosity;
  bulk_modulus = _bulk_modulus;
  gamma = _gamma;
  printf("viscosity: %f\n", viscosity);
  printf("gamma: %f\n", gamma);
  printf("bulk_modulus: %f\n", bulk_modulus);
  density = _density;
  name = "NewtonFluid";
  printf("density: %f\n", density);
}

void NewtonFluid::stress_update(ParticlesContainer &particles_ref, int mat_id) {
#ifdef CUDA_ENABLED
  KERNEL_STRESS_UPDATE_NEWTONFLUID<<<particles_ref.launch_config.tpb,
                                     particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.volumes_original_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.is_active_gpu.data()),
      particles_ref.num_particles, viscosity, bulk_modulus, gamma, mat_id);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    stress_update_newtonfluid(
        thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.volumes_original_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.is_active_gpu.data()), viscosity,
        bulk_modulus, gamma, mat_id, pid);
  }

#endif
}

NewtonFluid::~NewtonFluid() {}

} // namespace pyroclastmpm