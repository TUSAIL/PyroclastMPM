#include "pyroclastmpm/materials/linearelastic.cuh"

namespace pyroclastmpm
{

#ifdef CUDA_ENABLED
  extern Real __constant__ dt_gpu;
#else
  extern Real dt_cpu;
#endif

#include "linearelastic_inline.cuh"

  /**
   * @brief Construct a new Linear Elastic:: Linear Elastic object
   *
   * @param _density density of the material
   * @param _E young's modulus
   * @param _pois poissons ratio
   */
  LinearElastic::LinearElastic(const Real _density,
                               const Real _E,
                               const Real _pois)
  {
    E = _E;
    pois = _pois;
    bulk_modulus = (1. / 3.) * (E / (1. - 2. * pois));         // K
    shear_modulus = (1. / 2.) * E / (1 + pois);                // G
    lame_modulus = (pois * E) / ((1 + pois) * (1 - 2 * pois)); // lambda
    density = _density;
    name = "LinearElastic";
  }

  /**
   * @brief Compute the stress tensor for the material
   *
   * @param particles_ref particles container reference
   * @param mat_id material id
   */
  void LinearElastic::stress_update(ParticlesContainer &particles_ref,
                                    int mat_id)
  {

#ifdef CUDA_ENABLED
    KERNEL_STRESS_UPDATE_LINEARELASTIC<<<particles_ref.launch_config.tpb,
                                         particles_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
        particles_ref.num_particles, shear_modulus, lame_modulus, mat_id);

    gpuErrchk(cudaDeviceSynchronize());
#else
    for (int pid = 0; pid < particles_ref.num_particles; pid++)
    {
      update_linearelastic(
          particles_ref.stresses_gpu.data(),
          particles_ref.velocity_gradient_gpu.data(),
          particles_ref.volumes_gpu.data(),
          particles_ref.masses_gpu.data(),
          particles_ref.colors_gpu.data(),
          shear_modulus,
          lame_modulus,
          mat_id,
          pid);
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
  Real LinearElastic::calculate_timestep(Real cell_size, Real factor)
  {
    // https://www.sciencedirect.com/science/article/pii/S0045782520306885
    const Real c = sqrt((bulk_modulus + 4. * shear_modulus / 3.) / density);

    const Real delta_t = factor * (cell_size / c);

    printf("LinearElastic::calculate_timestep: %f", delta_t);
    return delta_t;
  }

  LinearElastic::~LinearElastic() {}

} // namespace pyroclastmpm