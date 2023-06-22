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
 * @file localrheo.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Local rheology material
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include "pyroclastmpm/materials/materials.h"

namespace pyroclastmpm {

/**
 * @brief This material is a hypoelastic material for local rheology
 * The implementation is based on the paper
 * Dunatunga, Sachith, and Ken Kamrin.
 * "Continuum modelling and simulation of granular flows through their many
 * phases." Journal of Fluid Mechanics 779 (2015): 483-513.
 */
class LocalGranularRheology : public Material {
public:
  /// @brief Construct a new Local Granular Rheology object
  /// @param _density material density
  /// @param _E Young's modulus
  /// @param _pois Poisson's ratio
  /// @param _I0 inertial number
  /// @param _mu_s critical friction angle (max)
  /// @param _mu_2 critical friction angle (min)
  /// @param _rho_c critical density
  /// @param _particle_diameter particle diameter
  /// @param _particle_density particle solid density
  LocalGranularRheology(const Real _density, const Real _E, const Real _pois,
                        const Real _I0, const Real _mu_s, const Real _mu_2,
                        const Real _rho_c, const Real _particle_diameter,
                        const Real _particle_density);

  /// @brief Destroy the Local Granular Rheology object
  ~LocalGranularRheology() final = default;

  /// @brief Perform stress update
  /// @param particles_ptr particles container
  /// @param mat_id material id
  void stress_update(ParticlesContainer &particles_ptr, int mat_id) override;

  /// @brief scalar composed of I0, particle diameter and particle density
  Real EPS;

  /// @brief static critical friction angle
  Real mu_s;

  /// @brief minimum critical friction angle
  Real mu_2;

  /// @brief critical density
  Real rho_c;

  /// @brief Inertial number
  Real I0;

  /// @brief Particle diameter
  Real particle_diameter;

  /// @brief Particle solid density
  Real particle_density;

  //// @brief Youngs modulus
  Real E;

  /// @brief Poisson's ratio
  Real pois;

  /// @brief Shear modulus (G)
  Real shear_modulus;

  /// @brief Lame modulus (lambda)
  Real lame_modulus;

  /// @brief Bulk modulus (K)
  Real bulk_modulus;
};

} // namespace pyroclastmpm