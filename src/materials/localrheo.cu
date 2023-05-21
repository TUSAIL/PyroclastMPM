

#include "pyroclastmpm/materials/localrheo.cuh"

namespace pyroclastmpm
{

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





__device__ __host__ inline double negative_root(double a, double b, double c)
{
  double x;
  if (b > 0)
  {
    x = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
  }
  else
  {
    x = (2 * c) / (-b + sqrt(b * b - 4 * a * c));
  }
  return x;
}

struct CalculateStress
{
  Real shear_modulus;
  Real lame_modulus;
  Real rho_c;
  Real mu_s;
  Real mu_2;
  Real I0;
  Real EPS;
  int mat_id;


  CalculateStress(
      const Real _shear_modulus,
      const Real _lame_modulus,
      const Real _rho_c,
      const Real _mu_s,
      const Real _mu_2,
      const Real _I0,
      const Real _EPS,
      const int _mat_id) : shear_modulus(_shear_modulus),
                           lame_modulus(_lame_modulus),
                           rho_c(_rho_c),
                           mu_s(_mu_s),
                           mu_2(_mu_2),
                           I0(_I0),
                           EPS(_EPS),
                           mat_id(_mat_id){};

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple tuple) const
    {
      Matrix3r &stress = thrust::get<0>(tuple);
      Matrixr vel_gradr = thrust::get<1>(tuple);
      Real volume = thrust::get<2>(tuple);
      Real mass = thrust::get<3>(tuple);
      Real color = thrust::get<4>(tuple);
      bool is_rigid = thrust::get<5>(tuple);
      
      if (is_rigid)
      {
        return;
      }
      if (color != mat_id) 
      {
        return;
      }

      #if DIM != 3
          Matrix3r vel_grad = Matrix3r::Zero();
          vel_grad.block(0, 0, DIM, DIM) = vel_gradr; // enforce plane strain
      #else
          const Matrix3r vel_grad = vel_gradr;
      #endif


      const Real rho = mass / volume;

      const Matrix3r vel_grad_T = vel_grad.transpose();

      const Matrix3r D = 0.5 * (vel_grad + vel_grad_T);

      const Matrix3r W = 0.5 * (vel_grad - vel_grad_T);

      const Matrix3r stress_prev = stress;


      // Jaunman rate
      const Matrix3r Gn = -stress_prev * W + W * stress_prev;

#ifdef CUDA_ENABLED
      const Matrix3r stress_trail =
          stress_prev +
          dt_gpu * (2 * shear_modulus * D +
                    lame_modulus * D.trace() * Matrix3r::Identity() + Gn);
#else
      const Matrix3r stress_trail =
          stress_prev +
          dt_cpu * (2 * shear_modulus * D +
                    lame_modulus * D.trace() * Matrix3r::Identity() + Gn);
#endif

      const Real pressure_trail = -(1. / (Real)3) * stress_trail.trace();

      const Matrix3r stress_trail_0 =
          stress_trail + pressure_trail * Matrix3r::Identity();

      // compute second invariant of the deviatoric stress
      const Matrix3r stress_trail_0_trans = stress_trail_0.transpose();

      const Real tau = sqrt(0.5 * (stress_trail_0 * stress_trail_0_trans).trace());


      // if ((pressure_trail < 0.0))
      if ((pressure_trail < 0.0) || (rho < rho_c))
      {
        stress = Matrix3r::Zero();
        return;
      }

      const Real S0 = mu_s * pressure_trail;
      
      Real tau_next, scale_factor;
      if (tau <= S0)
      {
        tau_next = tau;
        scale_factor = 1.0;
      }
      else
      {

        const Real S2 = mu_2 * pressure_trail;
        const Real GRAIN_RHO = 2450.0;
        const Real GRAIN_D = 0.005;
        #ifdef CUDA_ENABLED
        const Real alpha = shear_modulus * I0 * dt_gpu * sqrt(pressure_trail / GRAIN_RHO) / GRAIN_D;
        #else
        const Real alpha = shear_modulus * I0 * dt_cpu * sqrt(pressure_trail / GRAIN_RHO) / GRAIN_D;
        #endif

        
        const Real B = -(S2 + tau + alpha);

        const Real H = S2 * tau + S0 * alpha;
        tau_next = negative_root(1.0, B, H);
        scale_factor = tau_next / tau;
      }

      stress = scale_factor * stress_trail_0 - pressure_trail * Matrix3r::Identity();
      
    }
};




  /**
   * @brief call stress update procedure
   *
   * @param particles_ref particles container
   * @param mat_id material id
   */
  void LocalGranularRheology::stress_update(ParticlesContainer &particles_ref,
                                            int mat_id)
  {

    execution_policy exec;

    PARALLEL_FOR_EACH_ZIP(exec,
                      particles_ref.num_particles,
                      CalculateStress(shear_modulus,
                                      lame_modulus,
                                      rho_c,
                                      mu_s,
                                      mu_2,
                                      I0,
                                      EPS,
                                      mat_id),
                      particles_ref.stresses_gpu.begin(),
                      particles_ref.velocity_gradient_gpu.begin(),
                      particles_ref.volumes_gpu.begin(),
                      particles_ref.masses_gpu.begin(),
                      particles_ref.colors_gpu.begin(),
                      particles_ref.is_rigid_gpu.begin());

  }

  LocalGranularRheology::~LocalGranularRheology() {}

} // namespace pyroclastmpm