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

#include <thrust/execution_policy.h>

#include "pyroclastmpm/boundaryconditions/boundaryconditions.h"
#include "pyroclastmpm/common/helper.h"
#include "pyroclastmpm/common/output.h"
#include "pyroclastmpm/nodes/nodes.h"
#include "pyroclastmpm/spatialpartition/spatialpartition.h"

namespace pyroclastmpm {

/**
 * @brief Apply rigid particle boundary conditions
 *
 */
struct RigidBodyLevelSet : BoundaryCondition {
  // FUNCTIONS

  RigidBodyLevelSet(const Vectorr _COM = Vectorr::Zero(),
                    const cpu_array<int> _frames = {},
                    const cpu_array<Vectorr> _locations = {},
                    const cpu_array<Vectorr> _rotations = {},
                    const cpu_array<OutputType> _output_formats = {}

  );
  ~RigidBodyLevelSet(){};

  void initialize(NodesContainer &nodes_ref, ParticlesContainer &particles_ref);

  void calculate_grid_normals(NodesContainer &nodes_ref,
                              ParticlesContainer &particles_ref);

  void calculate_overlapping_rigidbody(NodesContainer &nodes_ref,
                                       ParticlesContainer &particles_ref);

  void set_velocities(ParticlesContainer &particles_ref);

  void set_position(ParticlesContainer &particles_ref);

  void apply_on_nodes_moments(NodesContainer &nodes_ref,
                              ParticlesContainer &particles_ref) override;

  // VARIABLES

  /** @brief number of animation frames for rigid body */
  int num_frames;

  /** @brief rigid body center of mass of previous step */
  Vectorr COM;

  /** @brief rigid body euler angles of previous step*/
  Vectorr ROT;

  /** @brief translational velocity of rigid body */
  Vectorr translational_velocity;

  /** @brief rotation matrix of rigid body */
  Matrixr rotation_matrix;

  /** @brief normals of non-rigid material points */
  gpu_array<Vectorr> normals_gpu;

  /** @brief flag if rigid grid node overlaps with non-rigid material point
  grid
   * nodes */
  gpu_array<bool> is_overlapping_gpu;

  /** @brief  closest rigid particle*/
  gpu_array<int> closest_rigid_particle_gpu;

  /** @brief animations frames (steps) */
  cpu_array<int> frames_cpu;

  /** @brief animations locations */
  cpu_array<Vectorr> locations_cpu;

  /** @brief animations euler angles */
  cpu_array<Vectorr> rotations_cpu;

  int current_frame = 0;
  cpu_array<OutputType> output_formats;

  Vectorr euler_angles;
  Vectorr angular_velocities;
};
} // namespace pyroclastmpm