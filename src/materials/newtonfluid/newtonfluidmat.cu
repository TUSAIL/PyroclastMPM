#include "pyroclastmpm/materials/newtonfluid/newtonfluidmat.cuh"

namespace pyroclastmpm {

NewtonFluid::NewtonFluid(const Real _density,
                         const Real _viscosity,
                         const Real _bulk_modulus,
                         const Real _gamma) {
  viscosity = _viscosity;
  bulk_modulus = _bulk_modulus;
  gamma = _gamma;
  density = _density;
  name = "NewtonFluid";
}

void NewtonFluid::stress_update(ParticlesContainer& particles_ref, int mat_id) {
  KERNEL_STRESS_UPDATE_NEWTONFLUID<<<particles_ref.launch_config.tpb,
                                     particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.pressures_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.volumes_original_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
      particles_ref.num_particles, viscosity, bulk_modulus, gamma, mat_id);
  gpuErrchk(cudaDeviceSynchronize());
}

NewtonFluid::~NewtonFluid() {}

}  // namespace pyroclastmpm