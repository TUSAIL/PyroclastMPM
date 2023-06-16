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

/**
 * [1] https://en.wikipedia.org/wiki/Infinitesimal_strain_theory
 */

__device__ __host__ inline void update_linearelastic(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_velocity_gradient_gpu,
    const Matrixr *particles_F_gpu, const Real *particles_masses_gpu,
    const uint8_t *particles_colors_gpu, bool *particles_is_active_gpu,
    const Real shear_modulus, const Real bulk_modulus, const int mat_id,
    const int tid) {

  const int particle_color = particles_colors_gpu[tid];
  if (!particles_is_active_gpu[tid]) {
    return;
  }

  if (particle_color != mat_id) {
    return;
  }

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

  const Matrixr vel_grad = particles_velocity_gradient_gpu[tid];
  Matrixr F = particles_F_gpu[tid]; // deformation gradient

  // (total strain current step) infinitesimal strain assumptions [1]
  const Matrixr deps_curr =
      0.5 * (vel_grad + vel_grad.transpose()) * dt; // pseudo strain rate

  const Matrixr eps_curr = 0.5 * (F.transpose() + F) - Matrixr::Identity();

  // hydrostatic stress and volumetric strain
  const Real eps_v_trail = eps_curr.trace();

  const Real p = bulk_modulus * eps_v_trail;

  // deviatoric stress (7.82) and strain eq (3.114) [2]
  const Matrixr eps_dev_trail =
      eps_curr - (1 / 3.) * eps_v_trail * Matrixr::Identity();

  const Matrixr dev_s = 2. * shear_modulus * eps_dev_trail;

  Matrix3r sigma = Matrix3r::Zero();
  sigma.block(0, 0, DIM, DIM) += dev_s;

  sigma += p * Matrix3r::Identity();

  particles_stresses_gpu[tid] = sigma;
}

#ifdef CUDA_ENABLED
__global__ void KERNEL_STRESS_UPDATE_LINEARELASTIC(
    Matrix3r *particles_stresses_gpu, Matrixr *particles_velocity_gradient_gpu,
    const Matrixr *particles_F_gpu, const Real *particles_masses_gpu,
    const uint8_t *particles_colors_gpu, bool *particles_is_active_gpu,
    const int num_particles, const Real bulk_modulus, const Real lame_modulus,
    const int mat_id) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  update_linearelastic(particles_stresses_gpu, particles_velocity_gradient_gpu,
                       particles_F_gpu, particles_masses_gpu,
                       particles_colors_gpu, particles_is_active_gpu,
                       shear_modulus, bulk_modulus, mat_id, tid);
}
#endif