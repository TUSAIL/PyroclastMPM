#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

__global__ void KERNEL_TRAIL_FE_DRUCKERPRAGER(
    Matrixr* particles_F_gpu,
    const Matrixr* particles_velocity_gradients_gpu,
    const uint8_t* particles_colors_gpu,
    const int num_particles,
    const int mat_id);
    
__global__ void KERNEL_STRESS_UPDATE_DRUCKERPRAGER(
    Matrix3r * particles_stresses_gpu,
    Matrixr* particles_F_gpu,
    Real * particles_logJp_gpu,
    Real * particles_pressure_gpu,
    const Matrixr * U_gpu,
    const Matrixr * V_gpu,
    const Vectorr *S_gpu,
    const uint8_t* particles_colors_gpu,
    const Real alpha,
    const Real shear_modulus,
    const Real lame_modulus,
    const Real cohesion,
    const Real vcs,
    const int num_particles,
    const int mat_id);

}