#include "pyroclastmpm/materials/vonmises.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern Real dt_cpu;
#endif

#include "vonmises_inline.h"

/**
 * @brief Construct a new Linear Elastic:: Linear Elastic object
 *
 * @param _density density of the material
 * @param _E young's modulus
 * @param _pois poissons ratio
 */
VonMises::VonMises(const Real _density, const Real _E, const Real _pois,
                   const Real _yield_stress, const Real _H) {
  E = _E;
  pois = _pois;
  bulk_modulus = (1. / 3.) * (E / (1. - 2. * pois));         // K
  shear_modulus = (1. / 2.) * E / (1 + pois);                // G
  lame_modulus = (pois * E) / ((1 + pois) * (1 - 2 * pois)); // lambda
  density = _density;

  yield_stress = _yield_stress;
  H = _H;

  name = "VonMises";

#if DIM != 3
  printf("VonMises material only implemented for 3D\n");
  exit(1);
#endif
}

void VonMises::initialize(ParticlesContainer &particles_ref, int mat_id) {
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
void VonMises::stress_update(ParticlesContainer &particles_ref, int mat_id) {

#ifdef CUDA_ENABLED
  KERNEL_STRESS_UPDATE_VONMISES<<<particles_ref.launch_config.tpb,
                                  particles_ref.launch_config.bpg>>>(
      // Matrix3r *particles_stresses_gpu,
      thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
      thrust::raw_pointer_cast(eps_e_gpu.data()),
      thrust::raw_pointer_cast(acc_eps_p_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.F_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.colors_gpu.data()), bulk_modulus,
      shear_modulus, yield_stress, H, mat_id, particles_ref.num_particles);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {

    update_vonmises(particles_ref.stresses_gpu.data(), eps_e_gpu.data(),
                    acc_eps_p_gpu.data(),
                    particles_ref.velocity_gradient_gpu.data(),
                    particles_ref.F_gpu.data(), particles_ref.colors_gpu.data(),
                    bulk_modulus, shear_modulus, yield_stress, H, mat_id, pid);
  }
#endif
}

/**
 * @brief Calculate the time step for the material
 *
 * @param cell_size grid cell size
 * @param factor factor to multiply the time step by
 * @return Real
 */
Real VonMises::calculate_timestep(Real cell_size, Real factor) {
  // https://www.sciencedirect.com/science/article/pii/S0045782520306885
  const Real c = sqrt((bulk_modulus + 4. * shear_modulus / 3.) / density);

  const Real delta_t = factor * (cell_size / c);

  printf("VonMises::calculate_timestep: %f", delta_t);
  return delta_t;
}

VonMises::~VonMises() {}

} // namespace pyroclastmpm