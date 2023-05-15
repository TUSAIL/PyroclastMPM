#include "pyroclastmpm/boundaryconditions/planardomain/planardomain_kernels.cuh"

namespace pyroclastmpm
{

    extern __constant__ Real dt_gpu;

    __global__ void KERNELS_APPLY_PLANARDOMAIN(
        Vectorr *particles_forces_external_gpu,
        const Vectorr *particles_positions_gpu,
        const Vectorr *particles_velocities_gpu,
        const Real *particles_volumes_gpu,
        const Real *particle_masses_gpu,
        const Vectorr axis0_friction,
        const Vectorr axis1_friction,
        const Vectorr domain_start,
        const Vectorr domain_end,
        const int num_particles)
    {
        const int mem_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (mem_index >= num_particles)
        {
            return;
        }

        const Vectorr pos = particles_positions_gpu[mem_index];
        const Vectorr vel = particles_velocities_gpu[mem_index];

        const Vectorr vel_norm = vel.normalized();

        const Real vol = particles_volumes_gpu[mem_index];

        const Real Radius = 0.5 * pow(vol, 1. / DIM);

        const Real mass = particle_masses_gpu[mem_index];

#if DIM == 3
        const Vectorr normals0[6] = {
            Vectorr({1, 0, 0}), Vectorr({0, 1, 0}), Vectorr({0, 0, 1})};
        const Vectorr normals1[6] = {
            Vectorr({-1, 0, 0}), Vectorr({0, -1, 0}), Vectorr({0, 0, -1})};

#elif DIM == 2
        const Vectorr normals0[2] = {
            Vectorr({1, 0}), Vectorr({0, 1})};
        const Vectorr normals1[2] = {
            Vectorr({-1, 0}), Vectorr({0, -1})};

#else
        const Vectorr normals0[1] = {Vectorr(1)};
        const Vectorr normals1[1] = {Vectorr(-1)};
#endif

        const Vectorr overlap0 = Vectorr::Ones() * Radius - (pos - domain_start);

        for (int i = 0; i < DIM; i++)
        {
            if (overlap0[i] > 0)
            {
                const Vectorr vel_depth = (overlap0[i] * normals0[i]).dot(vel) * normals0[i];
                const Vectorr fric_term = normals0[i] - axis0_friction(i) * (vel - normals0[i].dot(vel) * normals0[i]);
                particles_forces_external_gpu[mem_index] += (mass / pow(dt_gpu, 2.)) * overlap0[i] * fric_term;
            }
        }
        const Vectorr overlap1 = Vectorr::Ones() * Radius - (domain_end - pos);
        for (int i = 0; i < DIM; i++)
        {
            if (overlap1[i] > 0)
            {
                const Vectorr vel_depth = (overlap1[i] * normals1[i]).dot(vel) * normals1[i];
                const Vectorr fric_term = normals1[i] - axis1_friction(i) * (vel - normals1[i].dot(vel) * normals1[i]);
                particles_forces_external_gpu[mem_index] += (mass / pow(dt_gpu, 2.)) * overlap1[i] * fric_term;
            }
        }
    }

} // namespace pyroclastmpm