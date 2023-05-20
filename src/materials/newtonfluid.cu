#include "pyroclastmpm/materials/newtonfluid.cuh"

namespace pyroclastmpm {

NewtonFluid::NewtonFluid(const Real _density,
                         const Real _viscosity,
                         const Real _bulk_modulus,
                         const Real _gamma) {
  viscosity = _viscosity;
  bulk_modulus = _bulk_modulus;
  gamma = _gamma;
  density = _density;
  name = "NewtonFluid";
}

struct CalculateStress
{
  Real viscosity;
  Real bulk_modulus;
  Real gamma;
  int mat_id;
  CalculateStress(
    const Real _viscosity,
    const Real _bulk_modulus,
    const Real _gamma,
    const int _mat_id) : viscosity(_viscosity),
                          bulk_modulus(_bulk_modulus),
                          gamma(_gamma),
                          mat_id(_mat_id){};
      template <typename Tuple>
  __host__ __device__ void operator()(Tuple tuple) const
  {
      Matrix3r &stress = thrust::get<0>(tuple);
      const Matrixr vel_grad = thrust::get<1>(tuple);
      const Real mass = thrust::get<2>(tuple);
      const Real volume = thrust::get<3>(tuple);
      const Real volume_original = thrust::get<4>(tuple);
      const uint8_t color = thrust::get<5>(tuple);

      if (color != mat_id)
      {
          return;
      }

      const Matrixr vel_grad_T = vel_grad.transpose();
      const Matrixr strain_rate = 0.5 * (vel_grad + vel_grad_T);

      Matrixr deviatoric_part = strain_rate - (1. / 3) * strain_rate.trace() * Matrixr::Identity();

      const Real density = mass / volume;
      const Real density_original = mass / volume_original;
      const Real mu = density / density_original;

      Real pressure = bulk_modulus * (pow(mu, gamma) - 1);

      Matrixr cauchy_stress = 2 * viscosity * deviatoric_part - pressure * Matrixr::Identity();

#if DIM == 3
      stress = cauchy_stress;
#else
      stress.block(0, 0, DIM, DIM) = cauchy_stress;
#endif

  }
};

void NewtonFluid::stress_update(ParticlesContainer& particles_ref, int mat_id) {

    execution_policy exec;
    PARALLEL_FOR_EACH_ZIP(exec,
                          particles_ref.num_particles,
                          CalculateStress(viscosity,
                                          bulk_modulus,
                                          gamma,
                                          mat_id),
                          particles_ref.stresses_gpu.begin(),
                          particles_ref.velocity_gradient_gpu.begin(),
                          particles_ref.masses_gpu.data(),
                          particles_ref.volumes_gpu.data(),
                          particles_ref.volumes_original_gpu.data(),
                          particles_ref.colors_gpu.begin());
}

NewtonFluid::~NewtonFluid() {}

}  // namespace pyroclastmpm