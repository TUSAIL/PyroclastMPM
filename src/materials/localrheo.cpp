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

#include "pyroclastmpm/materials/localrheo.h"

namespace pyroclastmpm {

/**
 * @brief global step counter
 *
 */
extern int global_step_cpu;

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern Real dt_cpu;
#endif

#include "localrheo_inline.h"

/**
 * @brief Construct a new Local Granular Rheology:: Local Granular Rheology
 * object
 *
 * @param _density material density
 * @param _E  Young's modulus
 * @param _pois Poisson's ratio
 * @param _I0 inertial number
 * @param _mu_s static friction coefficient
 * @param _mu_2 dynamic friction coefficient
 * @param _rho_c critical density
 * @param _particle_diameter particle diameter
 * @param _particle_density particle density
 */
LocalGranularRheology::LocalGranularRheology(const Real _density, const Real _E,
                                             const Real _pois, const Real _I0,
                                             const Real _mu_s, const Real _mu_2,
                                             const Real _rho_c,
                                             const Real _particle_diameter,
                                             const Real _particle_density) {
  E = _E;
  pois = _pois;
  density = _density;

  bulk_modulus = (1. / 3.) * (E / (1. - 2. * pois));
  shear_modulus = (1. / 2.) * E / (1. + pois);
  lame_modulus = (pois * E) / ((1. + pois) * (1. - 2. * pois));

  I0 = _I0;
  mu_s = _mu_s;
  mu_2 = _mu_2;
  rho_c = _rho_c;
  particle_diameter = _particle_diameter;
  particle_density = _particle_density;

  EPS = I0 / sqrt(pow(particle_diameter, 2) * _particle_density);

  name = "LocalGranularRheology";
}

/**
 * @brief call stress update procedure
 *
 * @param particles_ref particles container
 * @param mat_id material id
 */
void LocalGranularRheology::stress_update(ParticlesContainer &particles_ref,
                                          int mat_id) {

#ifdef CUDA_ENABLED
  KERNEL_STRESS_UPDATE_LOCALRHEO<<<particles_ref.launch_config.tpb,
                                   particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.colors_gpu.data()), shear_modulus,
      lame_modulus, bulk_modulus, rho_c, mu_s, mu_2, I0, EPS,
      particles_ref.num_particles, mat_id);

  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {

    stress_update_localrheo(
        thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
        shear_modulus, lame_modulus, bulk_modulus, rho_c, mu_s, mu_2, I0, EPS,
        mat_id, pid);
  }

#endif
}

LocalGranularRheology::~LocalGranularRheology() {}

} // namespace pyroclastmpm