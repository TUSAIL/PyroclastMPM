// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern const Real dt_cpu;
#endif

/**
 * @brief Apply DEM planar domain boundary conditions
 *
 * The contact algorithm implemented is based on the one described in
 *
 * de Vaucorbeil, Alban, and Vinh Phu Nguyen.
 * "Modelling contacts with a total Lagrangian material point method."
 * Computer Methods in Applied Mechanics and Engineering 373 (2021): 113503.
 *
 * @param particles_forces_external_gpu external forces of particles
 * @param particles_positions_gpu positions of particles
 * @param particles_velocities_gpu velocities of particles
 * @param particles_volumes_gpu volumes of particles (updated)
 * @param particle_masses_gpu masses of particles
 * @param face0_friction friction angle for faces x0,y0,z0 (radians)
 * @param face1_friction friction angle for face x1,y1,z1 (radians)
 * @param domain_start start of domain
 * @param domain_end end of domain
 * @param mem_index index of particle in memory
 */
__host__ __device__ inline void apply_planardomain(
    Vectorr *particles_forces_external_gpu,
    const Vectorr *particles_positions_gpu,
    const Vectorr *particles_velocities_gpu, const Real *particles_volumes_gpu,
    const Real *particle_masses_gpu, const Vectorr face0_friction,
    const Vectorr face1_friction, const Grid grid, const int mem_index) {

  const Vectorr pos = particles_positions_gpu[mem_index];
  const Vectorr vel = particles_velocities_gpu[mem_index];

  const Real vol = particles_volumes_gpu[mem_index];

  const Real inv_dim = (Real)1. / (Real)DIM;

  const double pow_vol = pow(vol, inv_dim);

  const Real Radius = (Real)0.5 * (Real)pow_vol; // avoiding implicit casting

  const Real mass = particle_masses_gpu[mem_index];

  // print Radius mass and vol
  // printf("Radius: %f, mass: %f, vol: %f\n", Radius, mass, vol);

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

#if DIM == 3
  const Vectorr wall_normals0[6] = {Vectorr({1, 0, 0}), Vectorr({0, 1, 0}),
                               Vectorr({0, 0, 1})};
  const Vectorr wall_normals1[6] = {Vectorr({-1, 0, 0}), Vectorr({0, -1, 0}),
                               Vectorr({0, 0, -1})};

#elif DIM == 2
  const Vectorr wall_normals0[2] = {Vectorr({1, 0}), Vectorr({0, 1})};
  const Vectorr wall_normals1[2] = {Vectorr({-1, 0}), Vectorr({0, -1})};

#else
  const Vectorr wall_normals0[1] = {Vectorr(1)};
  const Vectorr wall_normals1[1] = {Vectorr(-1)};
#endif

// Radius - min distance of particle to wall
const Vectorr component_overlap0 = Vectorr::Ones() * Radius - (pos - grid.origin);

#pragma unroll
  for (int i = 0; i < DIM; i++) {
    const Real overlap = component_overlap0[i];
    const Vectorr normal = wall_normals0[i];
    if (overlap > 0) {

      const Vectorr vel_depth =
          (overlap * normal).dot(vel) * normal;
      const Vectorr fric_term =
          normal -
          face0_friction(i) * (vel - normal.dot(vel) * normal);
      particles_forces_external_gpu[mem_index] +=
          (mass / pow(dt, 2.)) * overlap * fric_term;

    }
  }
  const Vectorr component_overlap1 = Vectorr::Ones() * Radius - (grid.end - pos);

#pragma unroll
  for (int i = 0; i < DIM; i++) {
    const Real overlap = component_overlap1[i];
    const Vectorr normal = wall_normals1[i];
    if (overlap > 0) {


      const Vectorr vel_depth =
          (overlap * normal).dot(vel) * normal;
      const Vectorr fric_term =
          normal -
          face1_friction(i) * (vel -normal.dot(vel) * normal);
      particles_forces_external_gpu[mem_index] +=
          (mass / pow(dt, 2.)) * overlap * fric_term;
    }
  }
  
}

#ifdef CUDA_ENABLED
__global__ void KERNELS_APPLY_PLANARDOMAIN(
    Vectorr *particles_forces_external_gpu,
    const Vectorr *particles_positions_gpu,
    const Vectorr *particles_velocities_gpu, const Real *particles_volumes_gpu,
    const Real *particle_masses_gpu, const Vectorr face0_friction,
    const Vectorr face1_friction, const Grid grid, const int num_particles) {
  const int mem_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (mem_index >= num_particles) {
    return;
  }

  apply_planardomain(particles_forces_external_gpu, particles_positions_gpu,
                     particles_velocities_gpu, particles_volumes_gpu,
                     particle_masses_gpu, face0_friction, face1_friction, grid,
                     mem_index);
}

#endif

} // namespace pyroclastmpm