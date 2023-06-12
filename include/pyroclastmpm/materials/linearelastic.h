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

#pragma once

#include "pyroclastmpm/materials/materials.h"

namespace pyroclastmpm {

// class ParticlesContainer;  // Forward declarations

/**
 * @brief Linear elastic material
 *
 */
struct LinearElastic : Material {
  // FUNCTIONS

  /**
   * @brief Construct a new Linear Elastic material
   *
   * @param _E Young's modulus
   * @param _pois Poisson's ratio
   */
  LinearElastic(const Real _density, const Real _E, const Real _pois = 0.);

  ~LinearElastic();

  /**
   * @brief Perform stress update
   *
   * @param particles_ptr particles container
   * @param mat_id material id
   */
  void stress_update(ParticlesContainer &particles_ptr, int mat_id);

  Real calculate_timestep(Real cell_size, Real factor = 0.1) override;

  // VARIABLES

  /** @brief Youngs modulus */
  Real E;

  /** @brief Poisson's ratio */
  Real pois;

  /** @brief Shear modulus */
  Real shear_modulus;

  /** @brief Lame modulus */
  Real lame_modulus;

  /** @brief Bulk modulus */
  Real bulk_modulus;
};

} // namespace pyroclastmpm