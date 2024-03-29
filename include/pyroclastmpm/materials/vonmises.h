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
 * @file vonmises.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Von Mises material
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include "pyroclastmpm/materials/materials.h"

namespace pyroclastmpm {

/**
 * @brief Von Mises material
 * @details Small strain with isotropic linear strain hardening
 *
 *
 * Implementation based on
 * de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen.
 * Computational methods for plasticity: theory and applications.
 * John Wiley & Sons, 2011.
 *
 */
class VonMises : public Material {

public:
  /// @brief Construct a new Von Mises object
  /// @param _density material density (original)
  /// @param _E Young's modulus
  /// @param _pois Poisson's ratio
  /// @param _yield_stress initial yield stress
  /// @param _H hardening coefficient
  VonMises(const Real _density, const Real _E, const Real _pois,
           const Real _yield_stress, const Real _H);

  ~VonMises() final = default;

  /// @brief Perform stress update
  /// @param particles_ptr ParticlesContainer class
  /// @param mat_id material id
  void stress_update(ParticlesContainer &particles_ptr, int mat_id) override;

  /// @brief Calculate time step wave propagation speed
  /// @param cell_size Fell size of the background grid
  /// @param factor Scaling factor for speed
  /// @return Real a timestep
  Real calculate_timestep(Real cell_size, Real factor = (Real)0.1) override;

  /// @brief Initialize material (allocate memory for history variables)
  /// @param particles_ref ParticleContainer reference
  /// @param mat_id material id
  void initialize(const ParticlesContainer &particles_ref, int mat_id);

  /// @brief Initial yield stress sigma y_0
  Real yield_stress;

  /// @brief hardening coefficient H
  Real H;

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

  /// @brief elastic strain (infinitesimal)
  gpu_array<Matrixr> eps_e_gpu;

  /// @brief accumulated plastic strain (history) for hardening
  gpu_array<Real> acc_eps_p_gpu;

  /// @brief flag to update history or not
  /// @details useful for probing stress values during a single point
  /// integration test
  bool do_update_history = true;

  /// @brief flag to use velocity gradient instead of strain increment
  /// @details useful for single point integration test, to have deformation
  /// formulated as small strain
  bool is_velgrad_strain_increment = false;
};

} // namespace pyroclastmpm