

#include "pyroclastmpm/materials/localrheo.h"

namespace pyroclastmpm {

/**
 * @brief global step counter
 *
 */
extern int global_step_cpu;

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern Real dt_cpu;
#endif

#include "localrheo_inline.h"

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
 */
LocalGranularRheology::LocalGranularRheology(const Real _density, const Real _E,
                                             const Real _pois, const Real _I0,
                                             const Real _mu_s, const Real _mu_2,
                                             const Real _rho_c,
                                             const Real _particle_diameter,
                                             const Real _particle_density) {
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

  EPS = I0 / sqrt(pow(particle_diameter, 2) * _particle_density);

  name = "LocalGranularRheology";
}

/**
 * @brief call stress update procedure
 *
 * @param particles_ref particles container
 * @param mat_id material id
 */
void LocalGranularRheology::stress_update(ParticlesContainer &particles_ref,
                                          int mat_id) {

#ifdef CUDA_ENABLED
  KERNEL_STRESS_UPDATE_LOCALRHEO<<<particles_ref.launch_config.tpb,
                                   particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.colors_gpu.data()), shear_modulus,
      lame_modulus, bulk_modulus, rho_c, mu_s, mu_2, I0, EPS,
      particles_ref.num_particles, mat_id);

  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {

    stress_update_localrheo(
        thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
        shear_modulus, lame_modulus, bulk_modulus, rho_c, mu_s, mu_2, I0, EPS,
        mat_id, pid);
  }

#endif
}

LocalGranularRheology::~LocalGranularRheology() {}

} // namespace pyroclastmpm