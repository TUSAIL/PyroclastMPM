#pragma once

#include "pyroclastmpm/materials/localrheo/localrheomat_kernels.cuh"
#include "pyroclastmpm/materials/materials.cuh"

namespace pyroclastmpm
{

  class ParticlesContainer; // Forward declarations

  /**
   * @brief Local Granular Rheology material. This material is based on the mu(I)
   * rheology of Dunatunga et al. (2015).
   *
   */
  struct LocalGranularRheology : Material
  {
    /**
     * @brief Construct a new Local Granular Rheology object
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
     */
    LocalGranularRheology(const Real _density,
                          const Real _E,
                          const Real _pois,
                          const Real _I0,
                          const Real _mu_s,
                          const Real _mu_2,
                          const Real _rho_c,
                          const Real _particle_diameter,
                          const Real _particle_density);

    /**
     * @brief Destroy the Local Granular Rheology object
     *
     *
     */
    ~LocalGranularRheology();

    /**
     * @brief Perform stress update
     *
     * @param particles_ptr particles container
     * @param mat_id material id
     */
    void stress_update(ParticlesContainer &particles_ptr, int mat_id) override;

    void mp_benchmark(
      std::vector<Matrix3r> &_stress_cpu,
      std::vector<uint8_t> &_phases_cpu,
      const std::vector<Matrixr> _velocity_gradient_cpu,
      const std::vector<Real> _volume_cpu,
      const std::vector<Real> _mass_cpu);

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
  };

} // namespace pyroclastmpm