#include "pyroclastmpm/materials/localrheo/localrheomat_kernels.cuh"

namespace pyroclastmpm
{

  extern __constant__ Real dt_gpu;

  __device__ double negative_root(double a, double b, double c)
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

  __global__ void KERNEL_STRESS_UPDATE_LOCALRHEO(
      Matrix3r *particles_stresses_gpu,
      uint8_t *particles_phases_gpu,
      const Matrixr *particles_velocity_gradients_gpu,
      const Real *particles_volume_gpu,
      const Real *particles_mass_gpu,
      const uint8_t *particles_colors_gpu,
      const Real shear_modulus,
      const Real lame_modulus,
      const Real bulk_modulus,
      const Real rho_c,
      const Real mu_s,
      const Real mu_2,
      const Real I0,
      const Real EPS,
      const int num_particles,
      const int mat_id)
  {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_particles)
    {
      return;
    } // block access threads

    const int particle_color = particles_colors_gpu[tid];

    if (particle_color != mat_id)
    {
      return;
    }
    // particles_stresses_gpu[tid] += Matrix3r::Ones();

#if DIM != 3
    Matrix3r vel_grad = Matrix3r::Zero();
    vel_grad.block(0, 0, DIM, DIM) = particles_velocity_gradients_gpu[tid]; // enforce plane strain

#else
    const Matrix3r vel_grad = particles_velocity_gradients_gpu[tid];

#endif

    //  printf("vel_grad: %f %f %f %f %f %f %f %f %f \n", vel_grad(0,0), vel_grad(0,1), vel_grad(0,2), vel_grad(1,0), vel_grad(1,1), vel_grad(1,2), vel_grad(2,0), vel_grad(2,1), vel_grad(2,2));
    // Simplest case is when density is below a critical threshold
    const Real rho = particles_mass_gpu[tid] / particles_volume_gpu[tid];

    const Matrix3r vel_grad_T = vel_grad.transpose();

    const Matrix3r D = 0.5 * (vel_grad + vel_grad_T);

    const Matrix3r W = 0.5 * (vel_grad - vel_grad_T);

    const Matrix3r stress_prev = particles_stresses_gpu[tid];

    // Jaunman rate
    const Matrix3r Gn = -stress_prev * W + W * stress_prev;

    const Matrix3r stress_trail =
        stress_prev +
        dt_gpu * (2 * shear_modulus * D +
                  lame_modulus * D.trace() * Matrix3r::Identity() + Gn);

    const Real pressure_trail = -(1. / (Real)3) * stress_trail.trace();

    const Matrix3r stress_trail_0 =
        stress_trail + pressure_trail * Matrix3r::Identity();

    // particles_stresses_gpu[tid] = stress_trail_0;

    // compute second invariant of the deviatoric stress
    const Matrix3r stress_trail_0_trans = stress_trail_0.transpose();

    const Real tau = sqrt(0.5 * (stress_trail_0 * stress_trail_0_trans).trace());

    // if ((pressure_trail < 0.0))
    if ((pressure_trail < 0.0) || (rho < rho_c))
    {
      particles_stresses_gpu[tid] = Matrix3r::Zero();
      particles_phases_gpu[tid] = 2; // gass-like
      return;
    }

    const Real S0 = mu_s * pressure_trail;

    Real tau_next, scale_factor;
    int phase = 0;
    if (tau <= S0)
    {
      tau_next = tau;
      scale_factor = 1.0;
      phase = 0;
    }
    else
    {

      const Real S2 = mu_2 * pressure_trail;
      const Real GRAIN_RHO = 2450.0;
      const Real GRAIN_D = 0.005;
      const Real alpha = shear_modulus * I0 * dt_gpu * sqrt(pressure_trail / GRAIN_RHO) / GRAIN_D;
      const Real B = -(S2 + tau + alpha);

      const Real H = S2 * tau + S0 * alpha;
      tau_next = negative_root(1.0, B, H);
      scale_factor = tau_next / tau;
      phase = 1;
    }

    particles_stresses_gpu[tid] = scale_factor * stress_trail_0 - pressure_trail * Matrix3r::Identity();
    particles_phases_gpu[tid] = phase; // fluid-like
  }

} // namespace pyroclastmpm
