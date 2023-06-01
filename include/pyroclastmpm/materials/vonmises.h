#pragma once

#include "pyroclastmpm/materials/materials.h"

namespace pyroclastmpm {

// class ParticlesContainer;  // Forward declarations

/**
 * @brief Linear elastic material
 *
 */
struct VonMises : Material {
  // FUNCTIONS

  /**
   * @brief Construct a new Linear Elastic material
   *
   * @param _E Young's modulus
   * @param _pois Poisson's ratio
   */
  VonMises(const Real _density, const Real _E, const Real _pois,
           const Real _yield_stress, const Real _H);

  ~VonMises();

  /**
   * @brief Perform stress update
   *
   * @param particles_ptr particles container
   * @param mat_id material id
   */
  void stress_update(ParticlesContainer &particles_ptr, int mat_id);

  Real calculate_timestep(Real cell_size, Real factor = 0.1) override;

  void initialize(ParticlesContainer &particles_ref, int mat_id);

  // VARIABLES

  /* initial yield stress sigma y_0*/
  Real yield_stress;

  /* hardening coefficient H */
  Real H;

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

  /*! @brief elastic strain (infinitesimal) */
  gpu_array<Matrixr> eps_e_gpu;

  /*! @brief accumulated plastic strain (history) for hardening */
  gpu_array<Real> acc_eps_p_gpu;
};

} // namespace pyroclastmpm
