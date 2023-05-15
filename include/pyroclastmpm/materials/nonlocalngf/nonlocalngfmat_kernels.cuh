#pragma once

#include "pyroclastmpm/common/types_common.cuh"

// namespace pyroclastmpm {

// __global__ void KERNEL_STRESS_UPDATE_LOCALRHEO(
//     Matrix3r* particles_stresses_gpu,
//     Real* particles_pressure_gpu,
//     Real* particles_density_gpu,
//     int* particles_phases_gpu,
//     const Matrix3r* particles_velocity_gradients_gpu,
//     const Matrix3r* particles_F_gpu,
//     const Real* particles_volume_gpu,
//     const Real* particles_mass_gpu,
//     const int* particles_colors_gpu,
//     const Real shear_modulus,
//     const Real lame_modulus,
//     const Real bulk_modulus,
//     const Real rho_c,
//     const Real mu_s,
//     const Real mu_2,
//     const Real I0,
//     const Real EPS,
//     const Real original_density,
//     const int global_step,
//     const int num_particles,
//     const int mat_id);


// __global__ void KERNEL_STRESS_UPDATE_LOCALRHEO2(
//     Matrix3r* particles_stresses_gpu,
//     Real* particles_pressure_gpu,
//     Real* particles_density_gpu,
//     int* particles_phases_gpu,
//     const Matrix3r* particles_velocity_gradients_gpu,
//     const Matrix3r* particles_F_gpu,
//     const Real* particles_volume_gpu,
//     const Real* particles_mass_gpu,
//     const int* particles_colors_gpu,
//     const Real shear_modulus,
//     const Real lame_modulus,
//     const Real bulk_modulus,
//     const Real rho_c,
//     const Real mu_s,
//     const Real mu_2,
//     const Real I0,
//     const Real EPS,
//     const Real original_density,
//     const Real grain_density,
//     const Real grain_diameter,
//     const int global_step,
//     const int num_particles,
//     const int mat_id);
    
// }