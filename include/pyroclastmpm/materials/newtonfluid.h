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

/**
 * @brief Newtonian fluid material
 *
 */
struct NewtonFluid : Material {
  /**
   * @brief Construct a new Newton Fluid object
   *
   * @param _density material density
   * @param _viscocity material viscocity
   * @param _bulk_modulus bulk modulus
   * @param gamma gamma parameter
   */
  NewtonFluid(const Real _density, const Real _viscosity,
              const Real _bulk_modulus = 0., const Real _gamma = 7.);

  ~NewtonFluid();

  /**
   * @brief Compute the stress tensor
   *
   * @param particles_ptr pointer to the particles container
   * @param mat_id material id
   */
  void stress_update(ParticlesContainer &particles_ptr, int mat_id) override;

  /**
   * @brief viscocity of the fluid
   *
   */
  Real viscosity;

  /**
   * @brief bulk modulus of the fluid
   *
   */
  Real bulk_modulus;

  /**
   * @brief gamma parameter
   *
   */
  Real gamma;
};

} // namespace pyroclastmpm