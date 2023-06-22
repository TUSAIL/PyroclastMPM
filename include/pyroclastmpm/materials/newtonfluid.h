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
 * @file newtonfluid.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Newtonion fluid material
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include "pyroclastmpm/materials/materials.h"

namespace pyroclastmpm {

/**
 * @brief Newtonian fluid material
 * @details The implementation is based on the paper
 * de Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory,
 * implementation, and applications." Advances in applied mechanics 53 (2020):
 * 185-398. (Page 80)
 * It is important that global variables are set before the solver is
 * called.
 *
 * This can be done by calling the set_globals() function.
 *
 */
class NewtonFluid : public Material {
public:
  /// @brief Construct a new Newton Fluid object
  /// @param _density material density
  /// @param _viscocity material viscocity
  /// @param _bulk_modulus bulk modulus
  /// @param gamma gamma (7 for water and 1.4 for air)
  NewtonFluid(const Real _density, const Real _viscosity,
              const Real _bulk_modulus = 0., const Real _gamma = 7.);

  /// @brief Default constructor
  ~NewtonFluid() final = default;

  /// @brief Perform stress update
  /// @param particles_ptr ParticlesContainer class
  /// @param mat_id material id
  void stress_update(ParticlesContainer &particles_ptr, int mat_id) override;

  /// @brief viscocity of the fluid
  Real viscosity;

  /// @brief bulk modulus of the fluid
  Real bulk_modulus;

  /// @brief gamma parameter (7 for water and 1.4 for air)
  Real gamma;
};

} // namespace pyroclastmpm