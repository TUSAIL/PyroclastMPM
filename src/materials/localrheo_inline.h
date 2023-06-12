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

__device__ __host__ inline double negative_root(double a, double b, double c) {
  double x;
  if (b > 0) {
    x = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
  } else {
    x = (2 * c) / (-b + sqrt(b * b - 4 * a * c));
  }
  return x;
}

__device__ __host__ inline void stress_update_localrheo(
    Matrix3r *particles_stresses_gpu,
    const Matrixr *particles_velocity_gradients_gpu,
    const Real *particles_volume_gpu, const Real *particles_mass_gpu,
    const uint8_t *particles_colors_gpu, const Real shear_modulus,
    const Real lame_modulus, const Real bulk_modulus, const Real rho_c,
    const Real mu_s, const Real mu_2, const Real I0, const Real EPS,
    const int mat_id, const int tid) {

  const int particle_color = particles_colors_gpu[tid];

  if (particle_color != mat_id) {
    return;
  }
  // particles_stresses_gpu[tid] += Matrix3r::Ones();

#if DIM != 3
  Matrix3r vel_grad = Matrix3r::Zero();
  vel_grad.block(0, 0, DIM, DIM) =
      particles_velocity_gradients_gpu[tid]; // enforce plane strain

#else
  const Matrix3r vel_grad = particles_velocity_gradients_gpu[tid];
#endif

#ifdef CUDA_ENABLED
  const Real dt = dt_gpu;
#else
  const Real dt = dt_cpu;
#endif

  // Simplest case is when density is below a critical threshold
  const Real rho = particles_mass_gpu[tid] / particles_volume_gpu[tid];

  const Matrix3r vel_grad_T = vel_grad.transpose();

  const Matrix3r D = 0.5 * (vel_grad + vel_grad_T);

  const Matrix3r W = 0.5 * (vel_grad - vel_grad_T);

  const Matrix3r stress_prev = particles_stresses_gpu[tid];

  // Jaunman rate
  const Matrix3r Gn = -stress_prev * W + W * stress_prev;

  const Matrix3r stress_trail =
      stress_prev + dt * (2 * shear_modulus * D +
                          lame_modulus * D.trace() * Matrix3r::Identity() + Gn);

  const Real pressure_trail = -(1. / (Real)3) * stress_trail.trace();

  const Matrix3r stress_trail_0 =
      stress_trail + pressure_trail * Matrix3r::Identity();

  // compute second invariant of the deviatoric stress
  const Matrix3r stress_trail_0_trans = stress_trail_0.transpose();

  const Real tau = sqrt(0.5 * (stress_trail_0 * stress_trail_0_trans).trace());

  // if ((pressure_trail < 0.0))
  if ((pressure_trail < 0.0) || (rho < rho_c)) {
    particles_stresses_gpu[tid] = Matrix3r::Zero();
    return;
  }

  const Real S0 = mu_s * pressure_trail;

  Real tau_next, scale_factor;
  if (tau <= S0) {
    tau_next = tau;
    scale_factor = 1.0;
  } else {

    const Real S2 = mu_2 * pressure_trail;
    const Real GRAIN_RHO = 2450.0;
    const Real GRAIN_D = 0.005;
    const Real alpha =
        shear_modulus * I0 * dt * sqrt(pressure_trail / GRAIN_RHO) / GRAIN_D;
    const Real B = -(S2 + tau + alpha);

    const Real H = S2 * tau + S0 * alpha;
    tau_next = negative_root(1.0, B, H);
    scale_factor = tau_next / tau;
  }

  particles_stresses_gpu[tid] =
      scale_factor * stress_trail_0 - pressure_trail * Matrix3r::Identity();
}

#ifdef CUDA_ENABLED

__global__ void KERNEL_STRESS_UPDATE_LOCALRHEO(
    Matrix3r *particles_stresses_gpu,
    const Matrixr *particles_velocity_gradients_gpu,
    const Real *particles_volume_gpu, const Real *particles_mass_gpu,
    const uint8_t *particles_colors_gpu, const Real shear_modulus,
    const Real lame_modulus, const Real bulk_modulus, const Real rho_c,
    const Real mu_s, const Real mu_2, const Real I0, const Real EPS,
    const int num_particles, const int mat_id) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_particles) {
    return;
  } // block access threads

  stress_update_localrheo(
      particles_stresses_gpu, particles_velocity_gradients_gpu,
      particles_volume_gpu, particles_mass_gpu, particles_colors_gpu,
      shear_modulus, lame_modulus, bulk_modulus, rho_c, mu_s, mu_2, I0, EPS,
      mat_id, tid);
}

#endif