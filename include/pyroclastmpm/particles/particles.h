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

#include "pyroclastmpm/common/helper.h"
#include "pyroclastmpm/common/output.h"
#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/spatialpartition/spatialpartition.h"

namespace pyroclastmpm {

/*!
 * @brief Particles container class
 */
class ParticlesContainer {
public:
  // Tell the compiler to do what it would have if we didn't define actor:
  ParticlesContainer() = default;

  /*!
   * @brief Constructs Particle container class
   * @param _positions particle positions
   * @param _velocities particle velocities
   * @param _colors particle types (optional)
   * @param _is_rigid particle rigidness (optional)
   */
  ParticlesContainer(const cpu_array<Vectorr> &_positions,
                     const cpu_array<Vectorr> &_velocities = {},
                     const cpu_array<uint8_t> &_colors = {},
                     const cpu_array<bool> &_is_rigid = {}) noexcept;

  /**
   * @brief Resets the gpu arrays
   *
   * @param reset_psi if the node/particle shape functions should be reset
   */
  void reset(bool reset_psi = true);

  /*! @brief Reorder Particles arrays */
  void reorder();

  /*!
   * @brief calls the SpatialPartion class to partition particles into a
   * background grid
   */
  void partition();

  /*! @brief output vtk files */
  void output_vtk();

  /*! @brief calculate the particle volumes */
  void calculate_initial_volumes();

  /*! @brief calculate the particle masses */
  void calculate_initial_masses(int mat_id, Real density);

  /**
   * @brief Set the spatial partition of the particles
   *
   * @param start start of the spatial partition domain
   * @param end  end of the spatial partition domain
   * @param spacing cell size of the spatial partition
   */
  void set_spatialpartition(const Vectorr start, const Vectorr end,
                            const Real spacing);

  void spawn_particles();

  void set_spawner(int spawnRate, int spawnVolume);

  void set_output_formats(const std::vector<std::string> &_output_formats);

  /*! @brief particles' stresses, we always store 3x3 matrix for stresses
   */
  gpu_array<Matrix3r> stresses_gpu;

  /*! @brief particles' velocity gradients */
  gpu_array<Matrixr> velocity_gradient_gpu;

  /*! @brief particles' deformation matrices */
  gpu_array<Matrixr> F_gpu;

  /*! @brief particles' shape function gradients */
  gpu_array<Vectorr> dpsi_gpu;

  /*! @brief particles' coordinates */
  gpu_array<Vectorr> positions_gpu;

  /*! @brief particles' velocities */
  gpu_array<Vectorr> velocities_gpu;

  /*! @brief particles' velocities */
  gpu_array<Vectorr> forces_external_gpu;

  /*! @brief particles' masses */
  gpu_array<Real> masses_gpu;

  /*! @brief particles' volumes */
  gpu_array<Real> volumes_gpu;

  /*! @brief particles' volumes */
  gpu_array<Real> volumes_original_gpu;

  /*! @brief particles' shape functions */
  gpu_array<Real> psi_gpu;

  /*! @brief particles' colors (or material type) */
  gpu_array<uint8_t> colors_gpu;

  /*! @brief particles' colors (or material type) */
  gpu_array<bool> is_rigid_gpu;

  /*! @brief particles' active (or material type) */
  gpu_array<bool> is_active_gpu;

  /*! @brief spatial partitioning class */
  // creating temp object since we need domain size
  SpatialPartition spatial = SpatialPartition();

#ifdef CUDA_ENABLED
  GPULaunchConfig launch_config;
#endif

  std::vector<std::string> output_formats;

  /*! @brief Total Number of particles */
  int num_particles = 0;

  bool isRestart = false;

  int numColors = 0;

  int spawnRate = -1;

  int spawnIncrement = 0;

  int spawnVolume = 0;
};

} // namespace pyroclastmpm