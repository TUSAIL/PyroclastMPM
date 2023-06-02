#include "pyroclastmpm/particles/particles.h"

namespace pyroclastmpm {

// number of nodes surrounding each element
extern int num_surround_nodes_cpu;
extern int particles_per_cell_cpu;
extern int global_step_cpu;

/*!
 * @brief Constructs Particle container class
 * @param _positions particle positions
 * @param _velocities particle velocities
 * @param _colors particle types (optional)
 * @param _stresses particle stresses (optional)
 * @param _masses particle masses (optional)
 * @param _volumes particle volumes (optional)
 */
ParticlesContainer::ParticlesContainer(
    const cpu_array<Vectorr> _positions, const cpu_array<Vectorr> _velocities,
    const cpu_array<uint8_t> _colors, const cpu_array<bool> _is_rigid,
    const cpu_array<Matrix3r> _stresses, const cpu_array<Real> _masses,
    const cpu_array<Real> _volumes, const cpu_array<OutputType> _output_formats)
    : output_formats(_output_formats), spawnIncrement(0), spawnRate(-1),
      spawnVolume(0) {
  num_particles = _positions.size();

  set_default_device<Matrix3r>(num_particles, _stresses, stresses_gpu,
                               Matrix3r::Zero());
  set_default_device<Vectorr>(num_particles, _positions, positions_gpu,
                              Vectorr::Zero());
  set_default_device<Vectorr>(num_particles, _velocities, velocities_gpu,
                              Vectorr::Zero());
  set_default_device<uint8_t>(num_particles, _colors, colors_gpu, 0);
  set_default_device<bool>(num_particles, _is_rigid, is_rigid_gpu, false);

  set_default_device<bool>(num_particles, {}, is_active_gpu, true);

  set_default_device<Real>(num_particles, _masses, masses_gpu, -1.0);
  set_default_device<Real>(num_particles, _volumes, volumes_gpu, -1.0);

  set_default_device<Matrixr>(num_particles, {}, velocity_gradient_gpu,
                              Matrixr::Zero());
  set_default_device<Matrixr>(num_particles, {}, F_gpu, Matrixr::Identity());
  set_default_device<Vectorr>(num_surround_nodes_cpu * num_particles, {},
                              dpsi_gpu, Vectorr::Zero());
  set_default_device<Real>(num_particles, _volumes, volumes_original_gpu, -1.0);
  set_default_device<Real>(num_surround_nodes_cpu * num_particles, {}, psi_gpu,
                           0.0);

  set_default_device<Vectorr>(num_particles, {}, forces_external_gpu,
                              Vectorr::Zero());

  spatial = SpatialPartition(); // create a temporary partitioning object, since
                                // we are getting domain size

#ifdef CUDA_ENABLED
  launch_config.tpb = dim3(int((num_particles) / BLOCKSIZE) + 1, 1, 1);
  launch_config.bpg = dim3(BLOCKSIZE, 1, 1);
  gpuErrchk(cudaDeviceSynchronize());
#endif

  reset(); // reset needed
}

void ParticlesContainer::reset(bool reset_psi) {

  execution_policy exec;
  if (reset_psi) {
    thrust::fill(exec, psi_gpu.begin(), psi_gpu.end(), 0.);
    thrust::fill(exec, dpsi_gpu.begin(), dpsi_gpu.end(), Vectorr::Zero());
  }
  thrust::fill(exec, velocity_gradient_gpu.begin(), velocity_gradient_gpu.end(),
               Matrixr::Zero());
}

void ParticlesContainer::reorder() {
  // TODO fix reordering
  printf("Reorder not working correctly \n");
  reorder_device_array<Vectorr>(positions_gpu, spatial.sorted_index_gpu);
  reorder_device_array<Vectorr>(velocities_gpu, spatial.sorted_index_gpu);
  reorder_device_array<Matrix3r>(stresses_gpu, spatial.sorted_index_gpu);

  reorder_device_array<Matrixr>(velocity_gradient_gpu,
                                spatial.sorted_index_gpu);
  reorder_device_array<Matrixr>(F_gpu, spatial.sorted_index_gpu);
  reorder_device_array<Vectorr>(dpsi_gpu, spatial.sorted_index_gpu);
  reorder_device_array<uint8_t>(colors_gpu, spatial.sorted_index_gpu);
  reorder_device_array<bool>(is_rigid_gpu, spatial.sorted_index_gpu);
  reorder_device_array<Real>(volumes_gpu, spatial.sorted_index_gpu);
  reorder_device_array<Real>(volumes_original_gpu, spatial.sorted_index_gpu);
  reorder_device_array<Real>(masses_gpu, spatial.sorted_index_gpu);
  reorder_device_array<Real>(psi_gpu, spatial.sorted_index_gpu);
  reorder_device_array<Vectorr>(forces_external_gpu, spatial.sorted_index_gpu);
}

void ParticlesContainer::set_spawner(int _spawnRate, int _spawnVolume) {

  spawnRate = _spawnRate;
  spawnVolume = _spawnVolume;
  spawnIncrement = 0;
  execution_policy exec;
  thrust::fill(exec, is_active_gpu.begin(), is_active_gpu.end(), false);
}

void ParticlesContainer::spawn_particles() {
  if (spawnRate <= 0) {
    return;
  }
  if (global_step_cpu % spawnRate != 0) {
    return;
  }
  if (spawnIncrement >= num_particles) {
    return;
  }

  cpu_array<bool> is_active_cpu = is_active_gpu;

  for (int si = 0; si < spawnVolume; si++) {

    is_active_cpu[spawnIncrement + si] = true;
  }
  spawnIncrement += spawnVolume;

  is_active_gpu = is_active_cpu;
}

void ParticlesContainer::output_vtk() {

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

  cpu_array<Matrix3r> stresses_cpu = stresses_gpu;
  cpu_array<Matrixr> velocity_gradient_cpu = velocity_gradient_gpu;
  cpu_array<Matrixr> F_cpu = F_gpu;
  cpu_array<Vectorr> velocities_cpu = velocities_gpu;
  cpu_array<Vectorr> positions_cpu = positions_gpu;
  cpu_array<Real> masses_cpu = masses_gpu;
  cpu_array<Real> volumes_cpu = volumes_gpu;
  cpu_array<Real> volumes_original_cpu = volumes_original_gpu;

  cpu_array<int> colors_cpu = colors_gpu;

  cpu_array<int> is_rigid_cpu = is_rigid_gpu;

  for (int pi = 0; pi < num_particles; pi++) {

    is_rigid_cpu[pi] =
        !is_rigid_cpu[pi]; // flip to make sure we don't output rigid particles
  }

  bool exclude_rigid_from_output = false; // TODO make this an option?
  set_vtk_points(positions_cpu, polydata, is_rigid_cpu,
                 exclude_rigid_from_output);
  set_vtk_pointdata<int>(is_rigid_cpu, polydata, "isRigid", is_rigid_cpu,
                         exclude_rigid_from_output);
  set_vtk_pointdata<Vectorr>(positions_cpu, polydata, "Positions", is_rigid_cpu,
                             exclude_rigid_from_output);
  set_vtk_pointdata<Vectorr>(velocities_cpu, polydata, "Velocity", is_rigid_cpu,
                             exclude_rigid_from_output);
  set_vtk_pointdata<Matrix3r>(stresses_cpu, polydata, "Stress", is_rigid_cpu,
                              exclude_rigid_from_output);
  set_vtk_pointdata<Matrixr>(velocity_gradient_cpu, polydata,
                             "VelocityGradient", is_rigid_cpu,
                             exclude_rigid_from_output);
  set_vtk_pointdata<Matrixr>(F_cpu, polydata, "DeformationMatrix", is_rigid_cpu,
                             exclude_rigid_from_output);
  set_vtk_pointdata<Real>(masses_cpu, polydata, "Mass", is_rigid_cpu,
                          exclude_rigid_from_output);
  set_vtk_pointdata<Real>(volumes_cpu, polydata, "Volume", is_rigid_cpu,
                          exclude_rigid_from_output);
  set_vtk_pointdata<Real>(volumes_original_cpu, polydata, "VolumeOriginal",
                          is_rigid_cpu, exclude_rigid_from_output);
  set_vtk_pointdata<uint8_t>(colors_cpu, polydata, "Color", is_rigid_cpu,
                             exclude_rigid_from_output);

  // loop over output_formats
  for (auto format : output_formats) {
    write_vtk_polydata(polydata, "particles", format);
  }
}

void ParticlesContainer::set_spatialpartition(const Vectorr start,
                                              const Vectorr end,
                                              const Real spacing) {
  spatial = SpatialPartition(start, end, spacing, num_particles);
  partition();
}

void ParticlesContainer::partition() {
  spatial.calculate_hash(positions_gpu);

  spatial.sort_hashes();

  spatial.bin_particles();
};

void ParticlesContainer::calculate_initial_volumes() {
  if (isRestart) {
    return;
  }
  cpu_array<Real> volumes_cpu = volumes_gpu;
  for (int pi = 0; pi < num_particles; pi++) {

    if (volumes_cpu[pi] > 0.) {
      continue;
    }
#if DIM == 3
    volumes_cpu[pi] = spatial.cell_size * spatial.cell_size * spatial.cell_size;
#elif DIM == 2
    volumes_cpu[pi] = spatial.cell_size * spatial.cell_size;
#else
    volumes_cpu[pi] = spatial.cell_size;
#endif
    volumes_cpu[pi] /= particles_per_cell_cpu;
  }
  volumes_gpu = volumes_cpu;
  volumes_original_gpu = volumes_cpu;
}

void ParticlesContainer::calculate_initial_masses(int mat_id, Real density) {
  if (isRestart) {
    return;
  }
  cpu_array<int> colors_cpu = colors_gpu;
  cpu_array<Real> masses_cpu = masses_gpu;
  cpu_array<Real> volumes_cpu = volumes_gpu;

  for (int pi = 0; pi < num_particles; pi++) {
    if ((colors_cpu[pi] != mat_id) || (masses_cpu[pi] > 0.)) {
      continue;
    }

    masses_cpu[pi] = density * volumes_cpu[pi];
  }

  masses_gpu = masses_cpu;
}

} // namespace pyroclastmpm