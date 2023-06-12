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

#include "pyroclastmpm/materials/mohrcoulomb.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern Real dt_cpu;
#endif

#include "mohrcoulomb_inline.h"

/**
 * @brief Construct a new Linear Elastic:: Linear Elastic object
 *
 * @param _density density of the material
 * @param _E young's modulus
 * @param _pois poissons ratio
 */
MohrCoulomb::MohrCoulomb(const Real _density, const Real _E, const Real _pois,
                         const Real _cohesion, const Real _friction_angle,
                         const Real _dilatancy_angle, const Real _H) {
  E = _E;
  pois = _pois;
  bulk_modulus = (1. / 3.) * (E / (1. - 2. * pois));         // K
  shear_modulus = (1. / 2.) * E / (1 + pois);                // G
  lame_modulus = (pois * E) / ((1 + pois) * (1 - 2 * pois)); // lambda
  density = _density;

  cohesion = _cohesion;
  H = _H;
  friction_angle = _friction_angle * (PI / 180);
  dilatancy_angle = _dilatancy_angle * (PI / 180);
  name = "MohrCoulomb";

#if DIM != 3
  printf("MohrCoulomb material only implemented for 3D\n");
  exit(1);
#endif
}

void MohrCoulomb::initialize(ParticlesContainer &particles_ref, int mat_id) {
  // printf("thus runs inside associativevonmises\n");
  set_default_device<Real>(particles_ref.num_particles, {}, acc_eps_p_gpu, 0.0);
  set_default_device<Matrixr>(particles_ref.num_particles, {}, eps_e_gpu,
                              Matrixr::Zero());
}

/**
 * @brief Compute the stress tensor for the material
 *
 * @param particles_ref particles container reference
 * @param mat_id material id
 */
void MohrCoulomb::stress_update(ParticlesContainer &particles_ref, int mat_id) {

  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    update_mohrcoulomb(
        particles_ref.stresses_gpu.data(), eps_e_gpu.data(),
        acc_eps_p_gpu.data(), particles_ref.velocity_gradient_gpu.data(),
        particles_ref.F_gpu.data(), particles_ref.colors_gpu.data(),
        bulk_modulus, shear_modulus, cohesion, friction_angle, dilatancy_angle,
        H, mat_id, pid);
  }
  // #endif
}

/**
 * @brief Calculate the time step for the material
 *
 * @param cell_size grid cell size
 * @param factor factor to multiply the time step by
 * @return Real
 */
Real MohrCoulomb::calculate_timestep(Real cell_size, Real factor) {
  // https://www.sciencedirect.com/science/article/pii/S0045782520306885
  const Real c = sqrt((bulk_modulus + 4. * shear_modulus / 3.) / density);

  const Real delta_t = factor * (cell_size / c);

  printf("MohrCoulomb::calculate_timestep: %f", delta_t);
  return delta_t;
}

MohrCoulomb::~MohrCoulomb() {}

} // namespace pyroclastmpm