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

#include "pyroclastmpm/boundaryconditions/planardomain.h"

#include "planardomain_inline.h"

namespace pyroclastmpm {

/// @brief Construct a new object
/// @param face0_friction Friction angle (degrees) for cube face x0,y0,z0
/// @param face1_friction Friction angle (degrees)for cube face x1,y1,z1
PlanarDomain::PlanarDomain(Vectorr _face0_friction, Vectorr _face1_friction) {

  face0_friction = _face0_friction * (Real)(PI / 180.);
  face1_friction = _face1_friction * (Real)(PI / 180.);
}

/// @brief apply contact on particles
/// @param particles_ref ParticlesContainer reference
void PlanarDomain::apply_on_particles(ParticlesContainer &particles_ref) {

#ifdef CUDA_ENABLED
  KERNELS_APPLY_PLANARDOMAIN<<<particles_ref.launch_config.tpb,
                               particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.forces_external_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocities_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.masses_gpu.data()), face0_friction,
      face1_friction, particles_ref.spatial.grid.origin,
      particles_ref.spatial.grid.end, particles_ref.num_particles);

  gpuErrchk(cudaDeviceSynchronize());

#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    apply_planardomain(
        particles_ref.forces_external_gpu.data(),
        particles_ref.positions_gpu.data(), particles_ref.velocities_gpu.data(),
        particles_ref.volumes_gpu.data(), particles_ref.masses_gpu.data(),
        face0_friction, face1_friction, particles_ref.spatial.grid.origin,
        particles_ref.spatial.grid.end, pid);
  }
#endif
};

} // namespace pyroclastmpm