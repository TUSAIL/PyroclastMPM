#include "pyroclastmpm/boundaryconditions/planardomain/planardomain.cuh"

namespace pyroclastmpm
{

  PlanarDomain::PlanarDomain(Vectorr _axis0_friction, Vectorr _axis1_friction)
  {
    axis0_friction = _axis0_friction;
    axis1_friction = _axis1_friction;
  }

  void PlanarDomain::apply_on_particles(ParticlesContainer &particles_ref)
  {

    KERNELS_APPLY_PLANARDOMAIN<<<particles_ref.launch_config.tpb,
                                 particles_ref.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles_ref.forces_external_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.velocities_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
        thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
        axis0_friction,
        axis1_friction,
        particles_ref.spatial.grid_start,
        particles_ref.spatial.grid_end,
        particles_ref.num_particles);

    gpuErrchk(cudaDeviceSynchronize());
  };

}