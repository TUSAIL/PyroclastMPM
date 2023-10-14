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

/**
 * @file linearelastic.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Isotropic linear elastic material
 * @details This is a small strain implementation of the isotropic linear
 * elastic
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#include "pyroclastmpm/materials/linearelastic.h"

#include "linearelastic_inline.h"

namespace pyroclastmpm {

/// @brief Construct a new Linear Elastic material
/// @param _E Young's modulus
/// @param _pois Poisson's ratio
LinearElastic::LinearElastic(const Real _density, const Real _E,
                             const Real _pois)
    : E(_E), pois(_pois) {

  bulk_modulus = E / ((Real)3.0 * ((Real)1.0 - (Real)2.0 * pois));
  shear_modulus = E / ((Real)2.0 * ((Real)1 + pois));

  lame_modulus =
      E * pois / (((Real)1.0 + pois) * ((Real)1.0 - (Real)2.0 * pois));
  density = _density;
}

/// @brief Perform stress update
/// @param particles_ptr ParticlesContainer class
/// @param mat_id material id
void LinearElastic::stress_update(ParticlesContainer &particles_ref,
                                  int mat_id) {

#ifdef CUDA_ENABLED
  KERNEL_STRESS_UPDATE_LINEARELASTIC<<<particles_ref.launch_config.tpb,
                                       particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.F_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.is_active_gpu.data()),
      particles_ref.num_particles, lame_modulus, shear_modulus, bulk_modulus,
      mat_id);

  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    update_linearelastic(
        particles_ref.stresses_gpu.data(), particles_ref.F_gpu.data(),
        particles_ref.velocity_gradient_gpu.data(),
        particles_ref.colors_gpu.data(), particles_ref.is_active_gpu.data(),
        lame_modulus, shear_modulus, bulk_modulus, mat_id, pid);
  }
#endif
}

/**
 * @brief Calculate the time step for the material
 *
 * @param cell_size grid cell size
 * @param factor factor to multiply the time step by
 * @return Real
 */
Real LinearElastic::calculate_timestep(Real cell_size, Real factor) {
  // https://www.sciencedirect.com/science/article/pii/S0045782520306885
  const auto c = (Real)sqrt(
      (bulk_modulus + (Real)4. * shear_modulus / (Real)3.) / density);

  const Real delta_t = factor * (cell_size / c);

  printf("LinearElastic::calculate_timestep: %f", delta_t);
  return delta_t;
}

} // namespace pyroclastmpm