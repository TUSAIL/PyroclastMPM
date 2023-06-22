// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
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
 * @file linearelastic.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Isotropic linear elastic material
 * @details This is a small strain implementation of the isotropic linear
 * elastic
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include "pyroclastmpm/materials/materials.h"

namespace pyroclastmpm {

/**
 * @brief Isotropic linear elastic material
 * @details This is a small strain implementation of the isotropic linear.
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/common/global_settings.h"
 *        #include "pyroclastmpm/materials/linearelastic.h"
 *        #include "pyroclastmpm/particles/particles.h"
 *
 *        const std::vector<Matrixr> velgrad = {Matrixr::Identity() * 0.1};
 *        const std::vector<Matrixr> F = {(Matrixr::Identity() + velgrad * dt) *
 * Matrixr::Identity()}; particles.F_gpu = F;
 *
 *        set_global_dt(0.1);
 *
 *        auto particles = ParticlesContainer(pos);
 *
 *        auto mat = LinearElastic(1000, 0.1, 0.1);
 *
 *        particles.velocity_gradient_gpu = velgrad;
 *
 *        mat.stress_update(particles, 0);
 *
 *
 * \endverbatim
 *
 * Note the mat_id argument in the stress_update function. This is used to
 * select the correct material properties for the particles, in case of
 * multiple materials.
 */
class LinearElastic : public Material {
public:
  /// @brief Construct a new Linear Elastic material
  /// @param _E Young's modulus
  /// @param _pois Poisson's ratio
  LinearElastic(const Real _density, const Real _E, const Real _pois = 0.);

  /// @brief Destroy the Linear Elastic material
  ~LinearElastic() final = default;

  /// @brief Perform stress update
  /// @param particles_ptr ParticlesContainer class
  /// @param mat_id material id
  void stress_update(ParticlesContainer &particles_ptr, int mat_id) override;

  /// @brief Calculate time step wave propagation speed
  /// @param cell_size Fell size of the background grid
  /// @param factor Scaling factor for speed
  /// @return Real a timestep
  Real calculate_timestep(Real cell_size, Real factor = (Real)0.1) override;

  /// @brief Youngs modulus
  Real E;

  /// @brief Poisson's ratio
  Real pois;

  /// @brief Shear modulus
  Real shear_modulus;

  /// @brief Lame modulus
  Real lame_modulus;

  /// @brief Bulk modulus
  Real bulk_modulus;
};

} // namespace pyroclastmpm