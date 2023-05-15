

#include "pyroclastmpm/materials/localrheo/localrheomat.cuh"

namespace pyroclastmpm
{

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
   */
  LocalGranularRheology::LocalGranularRheology(const Real _density,
                                               const Real _E,
                                               const Real _pois,
                                               const Real _I0,
                                               const Real _mu_s,
                                               const Real _mu_2,
                                               const Real _rho_c,
                                               const Real _particle_diameter,
                                               const Real _particle_density)
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

    EPS = I0 / sqrt(pow(particle_diameter, 2) * _particle_density);

    name = "LocalGranularRheology";
  }

  void LocalGranularRheology::mp_benchmark(
      std::vector<Matrix3r> &_stress_cpu,
      std::vector<uint8_t> &_phases_cpu,
      const std::vector<Matrixr> _velocity_gradient_cpu,
      const std::vector<Real> _volume_cpu,
      const std::vector<Real> _mass_cpu)
  {
    const int num_particles = _stress_cpu.size();

    GPULaunchConfig launch_config;
    launch_config.tpb = dim3(int((num_particles) / BLOCKSIZE) + 1, 1, 1);
    launch_config.bpg = dim3(BLOCKSIZE, 1, 1);

    gpu_array<Matrix3r> stress_gpu;
    set_default_device<Matrix3r>(num_particles, _stress_cpu, stress_gpu, Matrix3r::Zero());
    gpu_array<Matrixr> velocity_gradient_gpu;
    set_default_device<Matrixr>(num_particles, _velocity_gradient_cpu, velocity_gradient_gpu, Matrixr::Zero());

    gpu_array<Real> volume_gpu, mass_gpu;
    set_default_device<Real>(num_particles, _volume_cpu, volume_gpu, 0.);
    set_default_device<Real>(num_particles, _mass_cpu, mass_gpu, 0.);

    gpu_array<uint8_t> phases_gpu, colors_gpu;
    set_default_device<uint8_t>(num_particles, _phases_cpu, phases_gpu, 0);
    set_default_device<uint8_t>(num_particles, {}, colors_gpu, 0);

    KERNEL_STRESS_UPDATE_LOCALRHEO<<<launch_config.tpb,
                                     launch_config.bpg>>>(
        thrust::raw_pointer_cast(stress_gpu.data()),
        thrust::raw_pointer_cast(phases_gpu.data()),
        thrust::raw_pointer_cast(velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(volume_gpu.data()),
        thrust::raw_pointer_cast(mass_gpu.data()),
        thrust::raw_pointer_cast(colors_gpu.data()),
        shear_modulus, lame_modulus, bulk_modulus, rho_c, mu_s, mu_2, I0, EPS, num_particles, 0);
    gpuErrchk(cudaDeviceSynchronize());

    // cpu_array<Matrix3r> stress_cpu = stress_gpu;
    // cpu_array<Real> phases_cpu = _phases_cpu;
    _stress_cpu = std::vector<Matrix3r>(stress_gpu.begin(), stress_gpu.end());
    _phases_cpu = std::vector<uint8_t>(phases_gpu.begin(), phases_gpu.end());
    ;
  }
  /**
   * @brief call stress update procedure
   *
   * @param particles_ref particles container
   * @param mat_id material id
   */
  void LocalGranularRheology::stress_update(ParticlesContainer &particles_ref,
                                            int mat_id)
  {

    KERNEL_STRESS_UPDATE_LOCALRHEO<<<particles_ref.launch_config.tpb,
                                     particles_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.phases_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
        shear_modulus, lame_modulus, bulk_modulus, rho_c, mu_s, mu_2, I0, EPS, particles_ref.num_particles, mat_id);
    // // printf("this runs ! \n ");
    // KERNEL_STRESS_UPDATE_LOCALRHEO<<<particles_ref.launch_config.tpb,
    //                                  particles_ref.launch_config.bpg>>>(
    //     thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
    //     thrust::raw_pointer_cast(particles_ref.pressures_gpu.data()),
    //     thrust::raw_pointer_cast(particles_ref.densities_gpu.data()),
    //     thrust::raw_pointer_cast(particles_ref.phases_gpu.data()),
    //     thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
    //     thrust::raw_pointer_cast(particles_ref.F_gpu.data()),
    //     thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
    //     thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
    //     thrust::raw_pointer_cast(particles_ref.colors_gpu.data()), shear_modulus,
    //     lame_modulus, bulk_modulus, rho_c, mu_s, mu_2, I0, EPS, density,
    //     global_step_cpu, particles_ref.num_particles, mat_id);

    gpuErrchk(cudaDeviceSynchronize());
  }

  LocalGranularRheology::~LocalGranularRheology() {}

} // namespace pyroclastmpm