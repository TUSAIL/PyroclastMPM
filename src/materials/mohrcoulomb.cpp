#include "pyroclastmpm/materials/mohrcoulomb.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern Real dt_cpu;
#endif

#include "mohrcoulomb_inline.h"

/**
 * @brief Construct a new Linear Elastic:: Linear Elastic object
 *
 * @param _density density of the material
 * @param _E young's modulus
 * @param _pois poissons ratio
 */
MohrCoulomb::MohrCoulomb(const Real _density, const Real _E, const Real _pois,
                         const Real _cohesion, const Real _friction_angle,
                         const Real _dilatancy_angle, const Real _H) {
  E = _E;
  pois = _pois;
  bulk_modulus = (1. / 3.) * (E / (1. - 2. * pois));         // K
  shear_modulus = (1. / 2.) * E / (1 + pois);                // G
  lame_modulus = (pois * E) / ((1 + pois) * (1 - 2 * pois)); // lambda
  density = _density;

  cohesion = _cohesion;
  H = _H;
  friction_angle = _friction_angle * (PI / 180);
  dilatancy_angle = _dilatancy_angle * (PI / 180);
  name = "MohrCoulomb";

#if DIM != 3
  printf("MohrCoulomb material only implemented for 3D\n");
  exit(1);
#endif
}

void MohrCoulomb::initialize(ParticlesContainer &particles_ref, int mat_id) {
  // printf("thus runs inside associativevonmises\n");
  set_default_device<Real>(particles_ref.num_particles, {}, acc_eps_p_gpu, 0.0);
  set_default_device<Matrixr>(particles_ref.num_particles, {}, eps_e_gpu,
                              Matrixr::Zero());
}

/**
 * @brief Compute the stress tensor for the material
 *
 * @param particles_ref particles container reference
 * @param mat_id material id
 */
void MohrCoulomb::stress_update(ParticlesContainer &particles_ref, int mat_id) {

  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    update_mohrcoulomb(
        particles_ref.stresses_gpu.data(), eps_e_gpu.data(),
        acc_eps_p_gpu.data(), particles_ref.velocity_gradient_gpu.data(),
        particles_ref.F_gpu.data(), particles_ref.colors_gpu.data(),
        bulk_modulus, shear_modulus, cohesion, friction_angle, dilatancy_angle,
        H, mat_id, pid);
  }
  // #endif
}

/**
 * @brief Calculate the time step for the material
 *
 * @param cell_size grid cell size
 * @param factor factor to multiply the time step by
 * @return Real
 */
Real MohrCoulomb::calculate_timestep(Real cell_size, Real factor) {
  // https://www.sciencedirect.com/science/article/pii/S0045782520306885
  const Real c = sqrt((bulk_modulus + 4. * shear_modulus / 3.) / density);

  const Real delta_t = factor * (cell_size / c);

  printf("MohrCoulomb::calculate_timestep: %f", delta_t);
  return delta_t;
}

MohrCoulomb::~MohrCoulomb() {}

} // namespace pyroclastmpm