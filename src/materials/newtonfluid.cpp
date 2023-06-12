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