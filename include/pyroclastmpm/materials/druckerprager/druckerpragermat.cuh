#pragma once

#include "pyroclastmpm/materials/druckerprager/druckerpragermat_kernels.cuh"
#include "pyroclastmpm/materials/materials.cuh"

#include <vector>

namespace pyroclastmpm
{

  class ParticlesContainer; // Forward declarations

  /**
   * @brief Drucker-Prager material model based on Gergely Klare (2015), AP Tampubolon (2017), Chuyuan
   * Fu (2018). Formulated in terms of finite strain.
   *
   */

  struct DruckerPrager : Material
  {
    /**
     * @brief Construct a new Drucker Prager:: Drucker Prager object
     *
     * @param _density material density
     * @param _E Young's modulus
     * @param _pois Poisson's ratio
     * @param _friction_angle
     * @param _cohesion Hardening parameter
     * @param _vcs volume correction scalar
     */
    DruckerPrager(const Real _density,
                  const Real _E,
                  const Real _pois,
                  const Real _friction_angle,
                  const Real _cohesion,
                  const Real _vcs);

    /**
     * @brief Destroy the Local Granular Rheology object
     *
     *
     */
    ~DruckerPrager();

    /**
     * @brief Perform stress update
     *
     * @param particles_ptr particles container
     * @param mat_id material id
     */
    void stress_update(ParticlesContainer &particles_ptr, int mat_id) override;

    /**
     * @brief outbound stress update for debugging version
     *
     * @param stress stress tensor
     * @param Fe elastic deformation gradient
     * @param logJp log of the determinant of the plastic deformation gradient (volume correction)
     * @param Fp_tr trail plastic deformation gradeint
     * @param alpha Hardening parameter
     * @param dim dimension
     */
    void outbound_stress_update(Matrix3r &stress,
                                Matrix3r &Fe,
                                Real &logJp,
                                const Matrix3r Fp_tr,
                                const Real alpha,
                                const int dim);

    /**
     * @brief Young's modulus
     */
    Real E;

    /**
     * @brief Poisson's ratio
     */
    Real pois;

    /**
     * @brief Shear modulus
     */
    Real shear_modulus;

    /**
     * @brief Lame modulus
     *
     */
    Real lame_modulus;

    /**
     * @brief Friction angle
     *
     */
    Real friction_angle;

    /**
     * @brief Cohesion parameter
     *
     */
    Real cohesion;

    /**
     * @brief Volume correction scalar
     *
     */
    Real vcs;

    /**
     * @brief related to friction angle
     *
     */
    Real alpha;
  };

} // namespace pyroclastmpm