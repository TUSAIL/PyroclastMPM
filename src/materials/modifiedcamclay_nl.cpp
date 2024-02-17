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

#include "pyroclastmpm/materials/modifiedcamclay_nl.h"

#include "modifiedcamclay_nl_inline.h"

namespace pyroclastmpm {

extern const int global_step_cpu;

/// @brief Construct a new Modified Cam Clay object
/// @param _density material density (original)
/// @param _E Young's modulus
/// @param _pois Poisson's ratio
/// @param _M Slope of critical state line
/// @param _lam slope of virgin consolidation line
/// @param _kap slope of swelling line
/// @param _Vs solid volume
/// @param R  preconsolidation ratio
/// @param _Pt Tensile yield hydrostatic stress
/// @param _beta Parameter related to size of outer diameter of ellipse
ModifiedCamClayNonLinear::ModifiedCamClayNonLinear(
    const Real _density, const Real _pois, const Real _M, const Real _lam,
    const Real _kap, const Real _Vs, const Real _R, const Real _Pt,
    const Real _beta, const Real _bulk_modulus)
    : M(_M), Pt(_Pt), beta(_beta), pois(_pois), lam(_lam), kap(_kap), Vs(_Vs),
      R(_R), bulk_modulus(_bulk_modulus) {

  shear_modulus = (3.0 * (1.0 - 2.0 * pois)) / (2.0 * (1.0 + pois));

  if (!std::isnan(bulk_modulus)) {
    shear_modulus *= bulk_modulus;
    printf("shear modulus %f bulk modulus %f \n", shear_modulus, bulk_modulus);
  } else {

    printf("bulk modulus not defined %f \n", bulk_modulus);
  }

  density = _density;
}

/// @brief Initialize material (allocate memory for history variables)
/// @param particles_ref ParticleContainer reference
/// @param mat_id material id
void ModifiedCamClayNonLinear::initialize(
    const ParticlesContainer &particles_ref, [[maybe_unused]] int mat_id) {

  set_default_device<Real>(particles_ref.num_particles, {}, alpha_gpu, 0.0);

  set_default_device<Real>(particles_ref.num_particles, {}, pc_gpu, 0.0);

  set_default_device<Matrixr>(particles_ref.num_particles, {}, eps_e_gpu,
                              Matrixr::Zero());

  set_default_device<Matrix3r>(particles_ref.num_particles,
                               particles_ref.stresses_gpu, stress_ref_gpu,
                               Matrix3r::Zero());

  cpu_array<Matrix3r> stresses_cpu = particles_ref.stresses_gpu;

  cpu_array<Real> pc_cpu = pc_gpu;

  cpu_array<Real> pressures_cpu =
      cpu_array<Real>(particles_ref.num_particles, 0.);

  // printf("hi! \n");
  for (int pi = 0; pi < particles_ref.num_particles; pi++) {
    // must be positive compression
    pressures_cpu[pi] = -(stresses_cpu[pi].trace() / 3.);

    const Real Pc0 = pressures_cpu[pi] * R;

    pc_cpu[pi] = Pc0;

    if (pressures_cpu[pi] > pc_cpu[pi]) {
      printf("ModifiedCamClayNonLinear::initialize: Warning: Pc0 (%f) > p0 "
             "(%f)  check "
             "R ( %f)\n",
             pc_cpu[pi], pressures_cpu[pi], R);
    } else {
      printf("ModifiedCamClayNonLinear::initialize: Pc0 (%f) > p0 (%f)  check "
             "R ( %f)\n",
             pc_cpu[pi], pressures_cpu[pi], R);
    }
  }
  pc_gpu = pc_cpu;
}

/// @brief Perform stress update
/// @param particles_ptr ParticlesContainer class
/// @param mat_id material id
void ModifiedCamClayNonLinear::stress_update(ParticlesContainer &particles_ref,
                                             int mat_id) {

#ifdef CUDA_ENABLED
  // TODO ADD KERNEL

  // KERNEL_STRESS_UPDATE_MCC_NL<<<particles_ref.launch_config.tpb,
  //                               particles_ref.launch_config.bpg>>>(
  //     thrust::raw_pointer_cast(particles_ref.stresses_gpu.data()),
  //     thrust::raw_pointer_cast(eps_e_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ref.volumes_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ref.volumes_original_gpu.data()),
  //     thrust::raw_pointer_cast(alpha_gpu.data()),
  //     thrust::raw_pointer_cast(pc_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ref.velocity_gradient_gpu.data()),
  //     thrust::raw_pointer_cast(particles_ref.colors_gpu.data()),
  //     thrust::raw_pointer_cast(stress_ref_gpu.data()), bulk_modulus,
  //     shear_modulus, M, lam, kap, Pt, beta, Vs, mat_id, do_update_history,
  //     is_velgrad_strain_increment, particles_ref.num_particles);

#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    update_modifiedcamclay_nl(
        particles_ref.stresses_gpu.data(), eps_e_gpu.data(),
        particles_ref.volumes_gpu.data(),
        particles_ref.volumes_original_gpu.data(), alpha_gpu.data(),
        pc_gpu.data(), particles_ref.velocity_gradient_gpu.data(),
        particles_ref.colors_gpu.data(), stress_ref_gpu.data(), bulk_modulus,
        shear_modulus, M, lam, kap, Pt, beta, Vs, mat_id, do_update_history,
        is_velgrad_strain_increment, pid);
  }
#endif
}

/// @brief Calculate time step wave propagation speed
/// @param cell_size Fell size of the background grid
/// @param factor Scaling factor for speed
/// @return Real a timestep
Real ModifiedCamClayNonLinear::calculate_timestep(Real cell_size, Real factor,
                                                  Real bulk_modulus,
                                                  Real shear_modulus,
                                                  Real density) {
  // https://www.sciencedirect.com/science/article/pii/S0045782520306885
  const auto c = (Real)sqrt((bulk_modulus + 4. * shear_modulus / 3.) / density);

  const Real delta_t = factor * (cell_size / c);

  printf("ModifiedCamClayNonLinear::calculate_timestep: %f", delta_t);
  return delta_t;
}

void ModifiedCamClayNonLinear::output_vtk(NodesContainer &nodes_ref,
                                          ParticlesContainer &particles_ref) {

  if (output_formats.empty()) {
    return;
  }

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

  cpu_array<Vectorr> positions_cpu = particles_ref.positions_gpu;
  cpu_array<Real> alpha_cpu = alpha_gpu;
  set_vtk_points(positions_cpu, polydata);
  set_vtk_pointdata<Real>(alpha_cpu, polydata, "alpha");

  for (const auto &format : output_formats) {
    write_vtk_polydata(polydata, "modified_cam_clay", format);
  }
}

} // namespace pyroclastmpm