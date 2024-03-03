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
 * @file particles.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief MPM particles class
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#include "pyroclastmpm/particles/particles.h"
#include "spdlog/spdlog.h"

namespace pyroclastmpm
{

  extern const int num_surround_nodes_cpu;
  extern const int particles_per_cell_cpu;
  extern const int global_step_cpu;

  /// @brief Constructs Particle container class
  /// @param _positions Particle positions (required)
  /// @param _velocities Particle velocities (optional)
  /// @param _colors Particle types (optional)
  /// @param _is_rigid mask of rigid particles (optional)
  ParticlesContainer::ParticlesContainer(
      const cpu_array<Vectorr> &_positions, const cpu_array<Vectorr> &_velocities,
      const cpu_array<uint8_t> &_colors,
      const cpu_array<Vectorr> &_rigid_positions,
      const cpu_array<Vectorr> &_rigid_velocities,
      const cpu_array<uint8_t> &_rigid_colors) noexcept
  {
    num_rigid_particles = static_cast<int>(_rigid_positions.size());

    int num_nonrigid_particles = static_cast<int>(_positions.size());

    num_particles = num_nonrigid_particles + num_rigid_particles;

    cpu_array<bool> is_rigid_cpu = cpu_array<bool>(num_particles, false);

    // combined rigid and nonrigid positions, velocities and colors
    cpu_array<Vectorr> combined_positions_cpu =
        cpu_array<Vectorr>(num_particles, Vectorr::Zero());

    cpu_array<Vectorr> combined_velocities_cpu =
        cpu_array<Vectorr>(num_particles, Vectorr::Zero());

    cpu_array<uint8_t> combined_colors_cpu = cpu_array<uint8_t>(num_particles, 0);

    for (int pi = 0; pi < num_particles; pi++)
    {
      if (pi < num_nonrigid_particles)
      {
        combined_positions_cpu[pi] = _positions[pi];

        if (_velocities.size() > 0)
        {
          combined_velocities_cpu[pi] = _velocities[pi];
        }
        if (_colors.size() > 0)
        {
          combined_colors_cpu[pi] = _colors[pi];
        }
      }
      else
      {
        int rigid_pid = pi - num_nonrigid_particles;
        combined_positions_cpu[pi] = _rigid_positions[rigid_pid];

        if (_rigid_velocities.size() > 0)
        {
          combined_velocities_cpu[pi] = _rigid_velocities[rigid_pid];
        }
        if (_rigid_colors.size() > 0)
        {
          combined_colors_cpu[pi] = _rigid_colors[rigid_pid];
        }

        is_rigid_cpu[pi] = true;
      }
    }

    total_memory_mb += set_default_device<bool>(num_particles, is_rigid_cpu, is_rigid_gpu, false);
    total_memory_mb += set_default_device<Vectorr>(num_particles, combined_positions_cpu,
                                                   positions_gpu, Vectorr::Zero());
    total_memory_mb += set_default_device<Vectorr>(num_particles, combined_velocities_cpu,
                                                   velocities_gpu, Vectorr::Zero());
    total_memory_mb += set_default_device<uint8_t>(num_particles, combined_colors_cpu, colors_gpu,
                                                   0);

    total_memory_mb += set_default_device<Matrix3r>(num_particles, {}, stresses_gpu,
                                                    Matrix3r::Zero());

    total_memory_mb += set_default_device<bool>(num_particles, {}, is_active_gpu, true);

    total_memory_mb += set_default_device<Real>(num_particles, {}, masses_gpu, -1.0);
    total_memory_mb += set_default_device<Real>(num_particles, {}, volumes_gpu, -1.0);

    total_memory_mb += set_default_device<Matrixr>(num_particles, {}, velocity_gradient_gpu,
                                                   Matrixr::Zero());

    total_memory_mb += set_default_device<Matrixr>(num_particles, {}, F_gpu, Matrixr::Identity());
    total_memory_mb += set_default_device<Vectorr>(num_surround_nodes_cpu * num_particles, {},
                                                   dpsi_gpu, Vectorr::Zero());
    total_memory_mb += set_default_device<Real>(num_particles, {}, volumes_original_gpu, -1.0);
    total_memory_mb += set_default_device<Real>(num_surround_nodes_cpu * num_particles, {}, psi_gpu,
                                                0.0);

    total_memory_mb += set_default_device<Vectorr>(num_particles, {}, forces_external_gpu,
                                                   Vectorr::Zero());

#ifdef CUDA_ENABLED
    launch_config.tpb = dim3(int((num_particles) / BLOCKSIZE) + 1, 1, 1);
    launch_config.bpg = dim3(BLOCKSIZE, 1, 1);
    gpuErrchk(cudaDeviceSynchronize());
#endif

    reset(); // reset needed
    
    spdlog::info("[Particles] Number of particles: {}", num_particles);
    spdlog::info("[Particles] Number of rigid particles: {}", num_rigid_particles);
    spdlog::info("[Particles] Total memory allocated: {:2f} MB", total_memory_mb);
    spdlog::info("[Particles] Memory allocated per particle: {:4f} MB", total_memory_mb / num_particles);
  }

  /// @brief Set output formats ("vtk","csv","obj")
  void ParticlesContainer::set_output_formats(
      const std::vector<std::string> &_output_formats)
  {
    output_formats = _output_formats;
  }

  /**
   * @brief Resets the gpu arrays
   * @param reset_psi If the node/particle shape functions should be reset
   */
  void ParticlesContainer::reset(bool reset_psi)
  {

    if (reset_psi)
    {
      thrust::fill(psi_gpu.begin(), psi_gpu.end(), 0.);
      thrust::fill(dpsi_gpu.begin(), dpsi_gpu.end(), Vectorr::Zero());
    }
    thrust::fill(velocity_gradient_gpu.begin(), velocity_gradient_gpu.end(),
                 Matrixr::Zero());

    thrust::fill(forces_external_gpu.begin(), forces_external_gpu.end(),
                 Vectorr::Zero());
  }

  /// @brief Reorder Particles arrays
  void ParticlesContainer::reorder()
  {
    // TODO fix reordering
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

  /// @brief Set the spawner rate and volume
  /// @param rate Rate of spawning (in steps)
  /// @param volume Volume (number of particles) to spawn
  void ParticlesContainer ::set_spawner(int _rate, int _volume)
  {
    spawner = SpawnerData(_rate, _volume);
    thrust::fill(is_active_gpu.begin(), is_active_gpu.end(), false);
  }

  /// @brief Output particle data
  /// @details Calls VTK helper functions located in `output.h`
  void ParticlesContainer::output_vtk() const
  {

    if (output_formats.empty())
    {
      return;
    }
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

    cpu_array<bool> is_rigid_cpu = is_rigid_gpu;
    cpu_array<bool> do_output_cpu = is_rigid_gpu;

    for (int pi = 0; pi < num_particles; pi++)
    {

      do_output_cpu[pi] =
          !is_rigid_cpu[pi]; // flip to make sure we don't output rigid particles
    }

    // post process data
    cpu_array<Real> p_post_cpu = cpu_array<Real>(num_particles, 0.);
    cpu_array<Real> q_post_cpu = cpu_array<Real>(num_particles, 0.);
    cpu_array<Real> mu_post_cpu = cpu_array<Real>(num_particles, 0.);
    cpu_array<Matrix3r> s_post_cpu =
        cpu_array<Matrix3r>(num_particles, Matrix3r::Zero());

    cpu_array<Real> dev_strain_cpu = cpu_array<Real>(num_particles, 0.);
    cpu_array<Real> vol_strain_cpu = cpu_array<Real>(num_particles, 0.);

    for (int pi = 0; pi < num_particles; pi++)
    {
      p_post_cpu[pi] = (stresses_cpu[pi].block(0, 0, DIM, DIM).trace() / 3.);

      // if (p_post_cpu[pi] > 0.) {
      //   printf("p_post_cpu[pi] > 0. %f\n %f %f %f", p_post_cpu[pi],
      //          stresses_cpu[pi](0, 0), stresses_cpu[pi](1, 1),
      //          stresses_cpu[pi](2, 2))
      // }
      s_post_cpu[pi] = stresses_cpu[pi] - Matrix3r::Identity() * p_post_cpu[pi];
      q_post_cpu[pi] = (Real)sqrt(
          3 * 0.5 * (s_post_cpu[pi] * s_post_cpu[pi].transpose()).trace());

      mu_post_cpu[pi] = abs(q_post_cpu[pi]) / (-p_post_cpu[pi]);

      Matrixr strain_tensor =
          0.5 * (F_cpu[pi].transpose() + F_cpu[pi]) - Matrixr::Identity();
      vol_strain_cpu[pi] = -strain_tensor.trace();
      Matrixr s_strain_tensor =
          strain_tensor + (1. / 3) * vol_strain_cpu[pi] * Matrixr::Identity();
      dev_strain_cpu[pi] = (Real)sqrt(
          0.5 * (s_strain_tensor * s_strain_tensor.transpose()).trace());

      // dev_strain_cpu[pi] =
    }

    set_vtk_points(positions_cpu, polydata, do_output_cpu,
                   exclude_rigid_from_output);
    set_vtk_pointdata<int>(is_rigid_cpu, polydata, "isRigid", do_output_cpu,
                           exclude_rigid_from_output);
    set_vtk_pointdata<Vectorr>(positions_cpu, polydata, "Positions",
                               do_output_cpu, exclude_rigid_from_output);
    set_vtk_pointdata<Vectorr>(velocities_cpu, polydata, "Velocity",
                               do_output_cpu, exclude_rigid_from_output);
    set_vtk_pointdata<Matrix3r>(stresses_cpu, polydata, "Stress", do_output_cpu,
                                exclude_rigid_from_output);
    set_vtk_pointdata<Matrixr>(velocity_gradient_cpu, polydata,
                               "VelocityGradient", do_output_cpu,
                               exclude_rigid_from_output);
    set_vtk_pointdata<Matrixr>(F_cpu, polydata, "DeformationMatrix",
                               do_output_cpu, exclude_rigid_from_output);
    set_vtk_pointdata<Real>(masses_cpu, polydata, "Mass", do_output_cpu,
                            exclude_rigid_from_output);
    set_vtk_pointdata<Real>(volumes_cpu, polydata, "Volume", do_output_cpu,
                            exclude_rigid_from_output);
    set_vtk_pointdata<Real>(volumes_original_cpu, polydata, "VolumeOriginal",
                            do_output_cpu, exclude_rigid_from_output);

    set_vtk_pointdata<uint8_t>(colors_cpu, polydata, "Color", do_output_cpu,
                               exclude_rigid_from_output);

    set_vtk_pointdata<Real>(p_post_cpu, polydata, "Pressure", do_output_cpu,
                            exclude_rigid_from_output);

    set_vtk_pointdata<Real>(q_post_cpu, polydata, "q", do_output_cpu,
                            exclude_rigid_from_output);

    set_vtk_pointdata<Real>(mu_post_cpu, polydata, "|q|/p", do_output_cpu,
                            exclude_rigid_from_output);

    set_vtk_pointdata<Real>(dev_strain_cpu, polydata, "dev strain", do_output_cpu,
                            exclude_rigid_from_output);

    set_vtk_pointdata<Real>(vol_strain_cpu, polydata, "vol strain", do_output_cpu,
                            exclude_rigid_from_output);

    // loop over output_formats
    for (const auto &format : output_formats)
    {
      write_vtk_polydata(polydata, "particles", format);
    }
  }

  /// @brief Set the spatial partition of the particles
  /// @param _grid Grid structure containing the grid information
  void ParticlesContainer::set_spatialpartition(const Grid &_grid)
  {
    spatial = SpatialPartition(_grid, num_particles);
    partition();
  }

  /// @brief Calls the SpatialPartion class to partition particles into a
  /// background grid
  void ParticlesContainer::partition()
  {
    spatial.calculate_hash(positions_gpu);

    spatial.sort_hashes();

    spatial.bin_particles();
  };

  /// @brief calculate the particle volumes
  /// @details Requires global variable particles_per_cell and to be
  /// @param particles_per_cell number of particles per cell
  /// @param cell_size cell size
  void ParticlesContainer::calculate_initial_volumes()
  {
    if (isRestart)
    {
      return;
    }
    cpu_array<Real> volumes_cpu = volumes_gpu;
    for (int pi = 0; pi < num_particles; pi++)
    {

      if (volumes_cpu[pi] > 0.)
      {
        continue;
      }

      volumes_cpu[pi] = (Real)pow(spatial.grid.cell_size, DIM);
      volumes_cpu[pi] /= (Real)particles_per_cell_cpu;
    }
    volumes_gpu = volumes_cpu;
    volumes_original_gpu = volumes_cpu;
  }

  /// @brief calculate the particle masses
  /// @details Requires global variable particles_per_cell and to be
  /// set using `set_global_particles_per_cell`
  /// @param mat_id material id
  /// @param density material density
  void ParticlesContainer::calculate_initial_masses(int mat_id, Real density)
  {
    if (isRestart)
    {
      return;
    }
    cpu_array<int> colors_cpu = colors_gpu;
    cpu_array<Real> masses_cpu = masses_gpu;
    cpu_array<Real> volumes_cpu = volumes_gpu;

    for (int pi = 0; pi < num_particles; pi++)
    {
      if ((colors_cpu[pi] != mat_id) || (masses_cpu[pi] > 0.))
      {
        continue;
      }

      masses_cpu[pi] = density * volumes_cpu[pi];
    }

    masses_gpu = masses_cpu;
  }

  void ParticlesContainer::spawn_particles()
  {
    if (spawner.rate <= 0)
    {
      return;
    }

    if (global_step_cpu % spawner.rate != 0)
    {
      return;
    }
    if (spawner.increment >= num_particles)
    {
      return;
    }

    cpu_array<bool> is_active_cpu = is_active_gpu;

    for (int si = 0; si < spawner.volume; si++)
    {

      is_active_cpu[spawner.increment + si] = true;
    }
    spawner.increment += spawner.volume;

    is_active_gpu = is_active_cpu;
  }

} // namespace pyroclastmpm