#include "pyroclastmpm/materials/linearelastic.cuh"

namespace pyroclastmpm
{

#ifdef CUDA_ENABLED
  extern Real __constant__ dt_gpu;
#else
  extern Real dt_cpu;
#endif

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

  struct CalculateStress
  {
    Real shear_modulus;
    Real lame_modulus;
    int mat_id;
    CalculateStress(
        const Real _shear_modulus,
        const Real _lame_modulus,
        const int _mat_id) : shear_modulus(_shear_modulus),
                             lame_modulus(_lame_modulus),
                             mat_id(_mat_id){};

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple tuple) const
    {
      Matrix3r &stress = thrust::get<0>(tuple);
      Matrixr velocity_gradient = thrust::get<1>(tuple);
      int color = thrust::get<2>(tuple);

      if (color != mat_id)
      {
        return;
      }

      const Matrixr vel_grad = velocity_gradient;
      const Matrixr velgrad_T = vel_grad.transpose();
      const Matrixr deformation_matrix = 0.5 * (vel_grad + velgrad_T);
#if CUDA_ENABLED
      const Matrixr strain_increments = deformation_matrix * dt_gpu;
#else
      const Matrixr strain_increments = deformation_matrix * dt_cpu;
#endif

#if DIM == 3
      Matrixr cauchy_stress = stress;
#else
      Matrix3r cauchy_stress_3d = stress;
      Matrixr cauchy_stress = cauchy_stress_3d.block(0, 0, DIM, DIM);
#endif

      cauchy_stress += lame_modulus * strain_increments *
                           Matrixr::Identity() +
                       2. * shear_modulus * strain_increments;
#if DIM == 3
      sigma = cauchy_stress;
#else
      cauchy_stress_3d.block(0, 0, DIM, DIM) = cauchy_stress;
      stress = cauchy_stress_3d;
#endif
    }
  };

  /**
   * @brief Compute the stress tensor for the material
   *
   * @param particles_ref particles container reference
   * @param mat_id material id
   */
  void LinearElastic::stress_update(ParticlesContainer &particles_ref,
                                    int mat_id)
  {

    execution_policy exec;

    PARALLEL_FOR_EACH_ZIP(exec,
                          particles_ref.num_particles,
                          CalculateStress(shear_modulus,
                                          lame_modulus,
                                          mat_id),
                          particles_ref.stresses_gpu.begin(),
                          particles_ref.velocity_gradient_gpu.begin(),
                          particles_ref.colors_gpu.begin());
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