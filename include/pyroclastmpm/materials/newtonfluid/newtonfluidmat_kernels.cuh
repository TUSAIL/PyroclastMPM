#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {


__global__ void KERNEL_STRESS_UPDATE_NEWTONFLUID(
    Matrix3r* particles_stresses_gpu,
    Real* particles_pressure_gpu,
    const Matrixr* particles_velocity_gradients_gpu,
    const Real* particles_masses_gpu,
    const Real* particles_volumes_gpu,
    const Real* particles_volumes_original_gpu,
    const uint8_t* particles_colors_gpu,
    const int num_particles,
    const Real viscocity,
    const Real bulk_modulus,
    const Real gamma,
    const int mat_id);

}
