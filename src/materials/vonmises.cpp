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

#include "pyroclastmpm/materials/vonmises.h"

#include "vonmises_inline.h"

namespace pyroclastmpm {

/// @brief Construct a new Von Mises object
/// @param _density material density (original)
/// @param _E Young's modulus
/// @param _pois Poisson's ratio
/// @param _yield_stress initial yield stress
/// @param _H hardening coefficient
VonMises::VonMises(const Real _density, const Real _E, const Real _pois,
                   const Real _yield_stress, const Real _H)
    : yield_stress(_yield_stress), H(_H), E(_E), pois(_pois) {

  bulk_modulus = E / ((Real)3.0 * ((Real)1.0 - (Real)2.0 * pois));
  shear_modulus = E / ((Real)2.0 * ((Real)1 + pois));

  density = _density;

#if DIM != 3
  printf("VonMises material only implemented for 3D\n");
  exit(1);
#endif
}

/// @brief Initialize material (allocate memory for history variables)
/// @param particles_ref ParticleContainer reference
/// @param mat_id material id
void VonMises::initialize(const ParticlesContainer &particles_ref,
                          [[maybe_unused]] int mat_id) {
  set_default_device<Real>(particles_ref.num_particles, {}, acc_eps_p_gpu, 0.0);
  set_default_device<Matrixr>(particles_ref.num_particles, {}, eps_e_gpu,
                              Matrixr::Zero());
}

/// @brief Perform stress update
/// @param particles_ptr ParticlesContainer class
/// @param mat_id material id
void VonMises::stress_update(ParticlesContainer &particles_ref, int mat_id) {

#ifdef CUDA_ENABLED
  KERNEL_STRESS_UPDATE_VONMISES<<<particles_ref.launch_config.tpb,
                                  particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
      thrust::raw_pointer_cast(eps_e_gpu.data()),
      thrust::raw_pointer_cast(acc_eps_p_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.colors_gpu.data()), bulk_modulus,
      shear_modulus, yield_stress, H, mat_id, do_update_history,
      is_velgrad_strain_increment, particles_ref.num_particles);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {

    update_vonmises(particles_ref.stresses_gpu.data(), eps_e_gpu.data(),
                    acc_eps_p_gpu.data(),
                    particles_ref.velocity_gradient_gpu.data(),
                    particles_ref.colors_gpu.data(), bulk_modulus,
                    shear_modulus, yield_stress, H, mat_id, do_update_history,
                    is_velgrad_strain_increment, pid);
  }
#endif
}

/// @brief Calculate time step wave propagation speed
/// @param cell_size Fell size of the background grid
/// @param factor Scaling factor for speed
/// @return Real a timestep
Real VonMises::calculate_timestep(Real cell_size, Real factor) {
  // https://www.sciencedirect.com/science/article/pii/S0045782520306885
  const auto c = (Real)sqrt((bulk_modulus + 4. * shear_modulus / 3.) / density);

  const Real delta_t = factor * (cell_size / c);

  printf("VonMises::calculate_timestep: %f", delta_t);
  return delta_t;
}

} // namespace pyroclastmpm