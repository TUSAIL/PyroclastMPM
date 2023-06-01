#include "pyroclastmpm/boundaryconditions/planardomain.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern Real dt_cpu;
#endif

#include "planardomain_inline.h"

PlanarDomain::PlanarDomain(Vectorr _axis0_friction, Vectorr _axis1_friction) {
  axis0_friction = _axis0_friction;
  axis1_friction = _axis1_friction;
}

struct ApplyPlanarDomain {
  Vectorr grid_start;
  Vectorr grid_end;
  Vectorr axis0_friction;
  Vectorr axis1_friction;
  ApplyPlanarDomain(const Vectorr _grid_start, const Vectorr _grid_end,
                    const Vectorr _axis0_friction,
                    const Vectorr _axis1_friction)
      : grid_start(_grid_start), grid_end(_grid_end),
        axis0_friction(_axis0_friction), axis1_friction(_axis1_friction){};

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple tuple) const {
    Vectorr &force_ext = thrust::get<0>(tuple);
    const Vectorr position = thrust::get<1>(tuple);
    const Vectorr velocity = thrust::get<2>(tuple);
    const Real volume = thrust::get<3>(tuple);
    Real mass = thrust::get<4>(tuple);

    // const Vectorr vel_norm = vel.normalized();

    const Real Radius = 0.5 * pow(volume, 1. / DIM);

#if DIM == 3
    const Vectorr normals0[6] = {Vectorr({1, 0, 0}), Vectorr({0, 1, 0}),
                                 Vectorr({0, 0, 1})};
    const Vectorr normals1[6] = {Vectorr({-1, 0, 0}), Vectorr({0, -1, 0}),
                                 Vectorr({0, 0, -1})};
#elif DIM == 2
    const Vectorr normals0[2] = {Vectorr({1, 0}), Vectorr({0, 1})};
    const Vectorr normals1[2] = {Vectorr({-1, 0}), Vectorr({0, -1})};
#else
    const Vectorr normals0[1] = {Vectorr(1)};
    const Vectorr normals1[1] = {Vectorr(-1)};
#endif

    const Vectorr overlap0 = Vectorr::Ones() * Radius - (position - grid_start);

#ifdef CUDA_ENABLED
    const Real dt = dt_gpu;
#else
    const Real dt = dt_cpu;
#endif
#pragma unroll
    for (int i = 0; i < DIM; i++) {
      if (overlap0[i] > 0) {
        const Vectorr vel_depth =
            (overlap0[i] * normals0[i]).dot(velocity) * normals0[i];
        const Vectorr fric_term =
            normals0[i] -
            axis0_friction(i) *
                (velocity - normals0[i].dot(velocity) * normals0[i]);
        force_ext += (mass / pow(dt, 2.)) * overlap0[i] * fric_term;
      }
    }
    const Vectorr overlap1 = Vectorr::Ones() * Radius - (grid_end - position);

#pragma unroll
    for (int i = 0; i < DIM; i++) {
      if (overlap1[i] > 0) {
        const Vectorr vel_depth =
            (overlap1[i] * normals1[i]).dot(velocity) * normals1[i];
        const Vectorr fric_term =
            normals1[i] -
            axis1_friction(i) *
                (velocity - normals1[i].dot(velocity) * normals1[i]);
        force_ext += (mass / pow(dt, 2.)) * overlap1[i] * fric_term;
      }
    }
  }
};

void PlanarDomain::apply_on_particles(ParticlesContainer &particles_ref) {

#ifdef CUDA_ENABLED
  KERNELS_APPLY_PLANARDOMAIN<<<particles_ref.launch_config.tpb,
                               particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.forces_external_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocities_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.masses_gpu.data()), axis0_friction,
      axis1_friction, particles_ref.spatial.grid_start,
      particles_ref.spatial.grid_end, particles_ref.num_particles);

  gpuErrchk(cudaDeviceSynchronize());

#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    apply_planardomain(
        particles_ref.forces_external_gpu.data(),
        particles_ref.positions_gpu.data(), particles_ref.velocities_gpu.data(),
        particles_ref.volumes_gpu.data(), particles_ref.masses_gpu.data(),
        axis0_friction, axis1_friction, particles_ref.spatial.grid_start,
        particles_ref.spatial.grid_end, pid);
  }
#endif
};

} // namespace pyroclastmpm