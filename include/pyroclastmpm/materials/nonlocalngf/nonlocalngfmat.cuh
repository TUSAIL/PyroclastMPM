// #pragma once

#include "pyroclastmpm/materials/nonlocalngf/nonlocalngfmat_kernels.cuh"
#include "pyroclastmpm/materials/materials.cuh"

namespace pyroclastmpm
{

  class ParticlesContainer; // Forward declarations

  /**
   * @brief Non local granular fluidity model based on Amin Haeri's work (2022)
   *
   */
  struct NonLocalNGF : Material
  {
    /**
     * @brief Construct a  new non local granular fluidity model
     *
     * @param _density material density
     * @param _E Young's modulus
     * @param _pois Poisson's ratio
     * @param _I0 inertial number
     * @param _mu_s critical friction angle (max)
     * @param _mu_2 critical friction angle (min)
     * @param _rho_c critical density
     * @param _particle_diameter particle diameter
     * @param _particle_density particle solid density
     * @param _A nonlocal amplitude
     */
    NonLocalNGF(const Real _density,
                const Real _E,
                const Real _pois,
                const Real _I0,
                const Real _mu_s,
                const Real _mu_2,
                const Real _rho_c,
                const Real _particle_diameter,
                const Real _particle_density,
                const Real _A);

    /**
     * @brief Destroy the Local Granular Rheology object
     *
     *
     */
    ~NonLocalNGF();

    /**
     * @brief Perform stress update
     *
     * @param particles_ptr particles container
     * @param mat_id material id
     */
    void stress_update(ParticlesContainer &particles_ptr, int mat_id) override;

    /**
     * @brief Calculated value
     */
    Real EPS;

    /**
     * @brief static critical friction angle
     *
     */
    Real mu_s;

    /**
     * @brief minimum critical friction angle
     *
     */
    Real mu_2;

    /**
     * @brief critical density
     *
     *
     */
    Real rho_c;

    /**
     * @brief Inertial number
     *
     */
    Real I0;

    /**
     * @brief Particle diameter
     *
     *
     */
    Real particle_diameter;

    /**
     * @brief Particle solid density
     *
     *
     */
    Real particle_density;

    /** @brief Youngs modulus */
    Real E;

    /** @brief Poisson's ratio */
    Real pois;

    /** @brief Shear modulus (G) */
    Real shear_modulus;

    /** @brief Lame modulus (lambda) */
    Real lame_modulus;

    /** @brief Bulk modulus (K) */
    Real bulk_modulus;

    /** @brief Non local amplitude (A=0 for local) */
    Real A;
  };

} // namespace pyroclastmpm