#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm
{

    extern __constant__ Real dt_gpu;

    /**
     * @brief Linear elastic stress update
     *
     * @param particles_stresses_gpu stress tensor of particles
     * @param particles_velocity_gradient_gpu velocity gradient of particles
     * @param particles_strain_increments strain increment of particles
     * @param particles_densities_gpu density of particles
     * @param particles_volume_gpu volume of particles
     * @param particles_masses_gpu mass of particles
     * @param particles_colors_gpu color of particles (material id)
     * @param num_particles number of particles
     * @param shear_modulus shear modulus of material
     * @param lame_modulus lame modulus of material
     * @param original_density original density of material
     * @param mat_id id of material
     */
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
        const int mat_id);

    // __global__ void KERNEL_STRESS_UPDATE_LINEARELASTIC_TLMPM(
    //     Matrix3r *particles_stresses_gpu,
    //     Matrixr *particles_velocity_gradient_gpu,
    //     Matrixr *particles_F_gpu,
    //     Real *particles_densities_gpu,
    //     const Real *particles_volumes_gpu,
    //     const Real *particles_masses_gpu,
    //     const uint8_t *particles_colors_gpu,
    //     const int num_particles,
    //     const Real shear_modulus,
    //     const Real lame_modulus,
    //     const int mat_id);
} // namespace pyroclastmpm