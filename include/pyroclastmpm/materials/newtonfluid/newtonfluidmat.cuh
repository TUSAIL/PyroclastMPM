#pragma once

#include "pyroclastmpm/materials/materials.cuh"
#include "pyroclastmpm/materials/newtonfluid/newtonfluidmat_kernels.cuh"

namespace pyroclastmpm
{

  /**
   * @brief Newtonian fluid material
   *
   */
  struct NewtonFluid : Material
  {
    /**
     * @brief Construct a new Newton Fluid object
     *
     * @param _density material density
     * @param _viscocity material viscocity
     * @param _bulk_modulus bulk modulus
     * @param gamma gamma parameter
     */
    NewtonFluid(const Real _density,
                const Real _viscosity,
                const Real _bulk_modulus = 0.,
                const Real _gamma = 7.);

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