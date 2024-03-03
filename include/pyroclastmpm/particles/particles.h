// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
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

#pragma once

/**
 * @file particles.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief MPM particles class
 * @details The main purpose of this class is to store particle data,
 * calculate volumes/masses, output, sort and partition particles.
 * Particles
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#include "pyroclastmpm/common/helper.h"
#include "pyroclastmpm/common/output.h"
#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/spatialpartition/spatialpartition.h"

namespace pyroclastmpm {

/**
 * @brief Container for MPM particles
 * @details This class contains particle data and relevant functions such as
 *  calculate volumes/masses, output,sort and partition particles. Particles
 * may have different material types (colors), be rigid or active.
 *
 * Rigid particles are not affected by the deformation of the background grid.
 * In order to use rigid particles, the user must specify the boundary condition
 * for rigid particles (`RigidParticleLevelSet`).
 *
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/particles/particles.h"
 *
 *        const std::vector<Vectorr> pos = {Vectorr({0.1, 0.1}), Vectorr({0.199,
 * 0.1}), Vectorr({0.82, 0.0}), Vectorr({0.82, 0.6})};
 *
 *        ParticlesContainer particles = ParticlesContainer(pos);
 *
 *        // A cell occupies a user-defined number of particles
 *        // NB: set the global array number of particles per cell
 *        set_global_particles_per_cell(2);
 *
 *       // Find the cell index of the particle within the background grid
 *       auto min = Vectorr::Zero();
 *       auto max = Vectorr::Ones();
 *       auto cell_size = 0.2;
 *       particles.set_spatialpartition(min, max, cell_size);
 *       particles.partition();
 *
 *       // Calculate the particle volumes and masses
 *       // Using number of particles per cell and cell size
 *
 *       particles.calculate_initial_volumes();
 *
 *       Real density = 1000;
 *       int mat_id = 0;
 *       particles.calculate_initial_masses(mat_id, density);
 *
 * \endverbatim
 *
 *
 */
class ParticlesContainer {

private:
  /// @brief Spawner data (data class) for spawning particles
  /// @details This class  is used internally and contains volume
  /// of particles that should be spawned per rate.
  __host__ __device__ struct SpawnerData {
    /// @brief Default constructor
    SpawnerData() = default;

    /// @brief Constructor
    /// @param rate Rate of spawning (in steps)
    /// @param volume Volume of particles to spawn
    SpawnerData(int rate, int volume) : rate(rate), volume(volume) {}

    /// @brief Rate of spawning particles (in steps)
    int rate = -1;

    /// @brief Volume of particles to spawn
    int volume = 0;

    /// @brief Increment counter
    int increment = 0;
  };

public:
  /// @brief Default constructor
  ParticlesContainer() = default;

  /// @brief Constructs Particle container class
  /// @param _positions Particle positions (required)
  /// @param _velocities Particle velocities (optional)
  /// @param _colors Particle types (optional)
  /// @param _is_rigid mask of rigid particles (optional)
  ParticlesContainer(const cpu_array<Vectorr> &_positions,
                     const cpu_array<Vectorr> &_velocities = {},
                     const cpu_array<uint8_t> &_colors = {},
                     const cpu_array<Vectorr> &_rigid_positions = {},
                     const cpu_array<Vectorr> &_rigid_velocities= {},
                     const cpu_array<uint8_t> &_rigid_colors = {}
                     ) noexcept;
  
  /**
   * @brief Resets the gpu arrays
   * @details Resets the gpu arrays to default values.
   * For the Total Lagrangian Formulation the shape functions
   * are not reset (reset_psi = false ).
   *
   * @param reset_psi If the node/particle shape functions should be reset
   */
  void reset(bool reset_psi = true);

  /// @brief Reorder Particles arrays
  void reorder();

  /// @brief Calls the SpatialPartion class to partition particles into a
  /// background grid
  void partition();

  /// @brief Output particle data
  /// @details Requires that `set_output_formats` is called first
  void output_vtk() const;

  /// @brief calculate the particle volumes
  /// @details Requires global variable particles_per_cell and to be
  /// set using `set_global_particles_per_cell`
  /// @param particles_per_cell number of particles per cell
  /// @param cell_size cell size
  void calculate_initial_volumes();

  /// @brief calculate the particle masses
  /// @details Requires global variable particles_per_cell and to be
  /// set using `set_global_particles_per_cell`
  /// @param mat_id material id
  /// @param density material density
  void calculate_initial_masses(int mat_id, Real density);

  /// @brief Set the spatial partition of the particles
  /// @param _grid Grid structure containing the grid information
  void set_spatialpartition(const Grid &_grid);

  /// @brief Set output formats ("vtk","csv","obj")
  void set_output_formats(const std::vector<std::string> &_output_formats);

  ///@brief Cauchy stress field of the particles
  ///@details We always consider plain strain, so the
  /// stress tensor is 3x3
  gpu_array<Matrix3r> stresses_gpu;

  /// @brief Velocity gradients of the particles
  gpu_array<Matrixr> velocity_gradient_gpu;

  /// @brief Deformation matrices of the particles
  gpu_array<Matrixr> F_gpu;

  /// @brief Gradients of the node/particle shape functions
  gpu_array<Vectorr> dpsi_gpu;

  /// @brief Coordinates of the particles
  gpu_array<Vectorr> positions_gpu;

  /// @brief Velocity fields of the particles
  gpu_array<Vectorr> velocities_gpu;

  /// @brief External forces applied on the particles
  gpu_array<Vectorr> forces_external_gpu;

  /// @brief Mass field of the particles
  gpu_array<Real> masses_gpu;

  /// @brief Volumes of the particles (updated)
  gpu_array<Real> volumes_gpu;

  /// @brief Volumes of the particles (initial)
  gpu_array<Real> volumes_original_gpu;

  /// @brief Shape functions of the node/particles
  gpu_array<Real> psi_gpu;

  /// @brief Material type (called colors) of the particles
  /// @details This is used when two `Materials` are used in
  /// the simulation.
  gpu_array<uint8_t> colors_gpu;

  /// @brief Flag for particles
  /// @details This prevents the particles
  /// from being updated in the stress update, P2G and G2P kernels.
  gpu_array<bool> is_rigid_gpu;

  /// @brief Additional flag if particles are active
  /// @details This prevents the particles updated and used
  /// in spawners
  gpu_array<bool> is_active_gpu;

  /// @brief Spatial partitioning of the particles
  /// @details This is used to efficiently access to bin
  /// particles in the grid and efficiently perform the P2G and G2P
  /// transfers.
  SpatialPartition spatial = SpatialPartition();

#ifdef CUDA_ENABLED
  GPULaunchConfig launch_config;
#endif

  std::vector<std::string> output_formats;

  /// @brief Total Number of particles (number of non-rigid + rigid particles)
  int num_particles = 0;

  /// @brief Number of rigid particles
  int num_rigid_particles = 0;


  /// @brief Option to exclude rigid particles from output
  bool exclude_rigid_from_output = true;

  /// @brief Flag if this is a restart simulations
  /// @details This is mainly used in the pickle functionality
  /// of Python bindings. This prevents the volume and mass from
  /// recalculating.
  bool isRestart = false;

  /// @brief Number of colors or material types
  int numColors = 0;

  /// @brief Spawns a certain `volume` of particles per `rate`
  void spawn_particles();

  /// @brief Set the spawner rate and volume
  /// @param rate Rate of spawning (in steps)
  /// @param volume Volume (number of particles) to spawn
  void set_spawner(int rate, int volume);

  /// @brief Spawner data (data class) for spawning particles
  SpawnerData spawner;

  double total_memory_mb = 0.0;
};

} // namespace pyroclastmpm