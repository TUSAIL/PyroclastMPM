
#include "pyroclastmpm/materials/nonlocalngf/nonlocalngfmat.cuh"

namespace pyroclastmpm {

/**
 * @brief global step counter
 *
 */
extern int global_step_cpu;

/**
 * @brief Construct a new Local Granular Rheology:: Local Granular Rheology
 * object
 *
 * @param _density material density
 * @param _E  Young's modulus
 * @param _pois Poisson's ratio
 * @param _I0 inertial number
 * @param _mu_s static friction coefficient
 * @param _mu_2 dynamic friction coefficient
 * @param _rho_c critical density
 * @param _particle_diameter particle diameter
 * @param _particle_density particle density
 * @param _A nonlocal amplitude
 */
NonLocalNGF::NonLocalNGF(const Real _density,
            const Real _E,
            const Real _pois,
            const Real _I0,
            const Real _mu_s,
            const Real _mu_2,
            const Real _rho_c,
            const Real _particle_diameter,
            const Real _particle_density,
            const Real _A ) 
            {
  E = _E;
  pois = _pois;
  density = _density;

  bulk_modulus = (1. / 3.) * (E / (1. - 2. * pois));
  shear_modulus = (1. / 2.) * E / (1. + pois);
  lame_modulus = (pois * E) / ((1. + pois) * (1. - 2. * pois));

  I0 = _I0;
  mu_s = _mu_s;
  mu_2 = _mu_2;
  rho_c = _rho_c;
  particle_diameter = _particle_diameter;
  particle_density = _particle_density;

  name = "NonLocalGranularRheology";

  /* what (additional) arrays to store*/

  // plastic deformation gradient
  // grad2_g
  // g
  // presure
  // mun


}


void do_batch_svd()
{
  //Do SVD of many small 3x3 matrices using CUSOLVER

  
}
/**
 * @brief call stress update procedure
 *
 * @param particles_ref particles container
 * @param mat_id material id
 */
void NonLocalNGF::stress_update(ParticlesContainer& particles_ref, int mat_id) {

}

NonLocalNGF::~NonLocalNGF() {}

}  // namespace pyroclastmpm