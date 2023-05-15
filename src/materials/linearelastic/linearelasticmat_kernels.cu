#include "pyroclastmpm/materials/linearelastic/linearelasticmat_kernels.cuh"

namespace pyroclastmpm
{

  extern __constant__ Real dt_gpu;

  __global__ void KERNEL_STRESS_UPDATE_LINEARELASTIC(
      Matrix3r *particles_stresses_gpu,
      Matrixr *particles_velocity_gradient_gpu,
      Matrixr *particles_strain_increments,
      Real *particles_densities_gpu,
      const Real *particles_volumes_gpu,
      const Real *particles_masses_gpu,
      const uint8_t *particles_colors_gpu,
      const int num_particles,
      const Real shear_modulus,
      const Real lame_modulus,
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

    const Matrixr vel_grad = particles_velocity_gradient_gpu[tid];
    const Matrixr velgrad_T = vel_grad.transpose();
    const Matrixr deformation_matrix = 0.5 * (vel_grad + velgrad_T);
    const Matrixr strain_increments = deformation_matrix * dt_gpu;

#if DIM == 3
    Matrixr cauchy_stress = particles_stresses_gpu[tid];
#else
    Matrix3r cauchy_stress_3d = particles_stresses_gpu[tid];
    Matrixr cauchy_stress = cauchy_stress_3d.block(0, 0, DIM, DIM);
#endif

    cauchy_stress += lame_modulus * strain_increments *
                         Matrixr::Identity() +
                     2. * shear_modulus * strain_increments;
#if DIM == 3
    particles_stresses_gpu[tid] = cauchy_stress;
#else
    cauchy_stress_3d.block(0, 0, DIM, DIM) = cauchy_stress;
    particles_stresses_gpu[tid] = cauchy_stress_3d;
#endif

  }

//     __global__ void KERNEL_STRESS_UPDATE_LINEARELASTIC_TLMPM(
//       Matrix3r *particles_stresses_gpu,
//       Matrixr *particles_velocity_gradient_gpu,
//       Matrixr *particles_F_gpu,
//       Real *particles_densities_gpu,
//       const Real *particles_volumes_gpu,
//       const Real *particles_masses_gpu,
//       const uint8_t *particles_colors_gpu,
//       const int num_particles,
//       const Real shear_modulus,
//       const Real lame_modulus,
//       const int mat_id)
//   {
//     const int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     if (tid >= num_particles)
//     {
//       return;
//     } // block access threads

//     const int particle_color = particles_colors_gpu[tid];

//     if (particle_color != mat_id)
//     {
//       return;
//     }

//     // const Matrixr vel_grad = particles_velocity_gradient_gpu[tid];
//     // const Matrixr velgrad_T = vel_grad.transpose();
//     // const Matrixr deformation_matrix = 0.5 * (vel_grad + velgrad_T);
//     const Matrixr F = particles_F_gpu[tid];
//     const Matrixr strain_increments = F * dt_gpu;

// #if DIM == 3
//     Matrixr cauchy_stress = particles_stresses_gpu[tid];
// #else
//     Matrix3r cauchy_stress_3d = particles_stresses_gpu[tid];
//     Matrixr cauchy_stress = cauchy_stress_3d.block(0, 0, DIM, DIM);
// #endif

//     cauchy_stress += lame_modulus * strain_increments *
//                          Matrixr::Identity() +
//                      2. * shear_modulus * strain_increments;
// #if DIM == 3
//     particles_stresses_gpu[tid] = cauchy_stress;
// #else
//     cauchy_stress_3d.block(0, 0, DIM, DIM) = cauchy_stress;
//     particles_stresses_gpu[tid] = cauchy_stress_3d;
// #endif

//   }
} // namespace pyroclastmpm