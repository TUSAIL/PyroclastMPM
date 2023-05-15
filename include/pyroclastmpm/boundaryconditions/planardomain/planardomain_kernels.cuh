#pragma once

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm {

    __global__ void KERNELS_APPLY_PLANARDOMAIN(
        Vectorr *particles_forces_external_gpu,
        const Vectorr *particles_positions_gpu,
        const Vectorr *particles_velocities_gpu,
        const Real *particle_volumes_gpu,
        const Real *particle_masses_gpu,
        const Vectorr axis0_friction,
        const Vectorr axis1_friction,
        const Vectorr domain_start,
        const Vectorr domain_end,
        const int num_particles);

}