#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

__global__ void KERNEL_STRESS_UPDATE_LOCALRHEO(
    Matrix3r* particles_stresses_gpu,
    uint8_t* particles_phases_gpu,
    const Matrixr* particles_velocity_gradients_gpu,
    const Real* particles_volume_gpu,
    const Real* particles_mass_gpu,
    const uint8_t* particles_colors_gpu,
    const Real shear_modulus,
    const Real lame_modulus,
    const Real bulk_modulus,
    const Real rho_c,
    const Real mu_s,
    const Real mu_2,
    const Real I0,
    const Real EPS,
    const int num_particles,
    const int mat_id);


    
}