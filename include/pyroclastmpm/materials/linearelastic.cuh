#pragma once

#include "pyroclastmpm/materials/materials.cuh"

namespace pyroclastmpm
{

  // class ParticlesContainer;  // Forward declarations

  /**
   * @brief Linear elastic material
   *
   */
  struct LinearElastic : Material
  {
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
