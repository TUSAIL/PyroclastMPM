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

/**
 * @file .h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Rigid body level set
 *
 * @version 0.1
 * @date 2023-06-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <thrust/execution_policy.h>

#include "pyroclastmpm/boundaryconditions/boundaryconditions.h"
#include "pyroclastmpm/common/helper.h"
#include "pyroclastmpm/common/output.h"
#include "pyroclastmpm/nodes/nodes.h"
#include "pyroclastmpm/spatialpartition/spatialpartition.h"

namespace pyroclastmpm {

/**
 * @brief Apply rigid body level set boundary conditions
 * @details A rigid body consists of a set of rigid particles that are
 * connected. These rigid particles are defined by a mask in ParticlesContainer
 *
 * Tools are helpful convert STL files to rigid particles.
 *
 * A motion .chan file can be parsed to input the animation frames. For more
 * information on .chan files see
 * https://docs.blender.org/manual/en/latest/addons/import_export/anim_nuke_chan.html
 *
 *
 * \verbatim embed:rst:leading-asterisk
 *     Example usage (constant)
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/boundaryconditions/rigidbody.h"
 *
 *        // set globals
 *
 *        // Be sure is_rigid mask in ParticlesContainer is set
 *
 *        // Create a static rigid body
 *        rigid_body_level_set = RigidBodyLevelSet();
 *
 *        // or
 *
 *        // Create a rigid body with animation
 *        rigid_body_level_set = RigidBodyLevelSet(
 *                             com_vector,
 *                             frames_vector,
 *                             locations_vector,
 *                              rotations_vector);
 *
 *        // Add rigidbody boundary condition to simulation
 *
 * \endverbatim
 *
 *
 */
class RigidBodyLevelSet : public BoundaryCondition {
public:
  /// @brief Construct a new Rigid Body Level Set object
  /// @param _COM center of mass of rigid body
  /// @param _frames animation frames
  /// @param _locations animation locations
  /// @param _rotations animation rotations
  RigidBodyLevelSet(const Vectorr _COM = Vectorr::Zero(),
                    const cpu_array<int> &_frames = {},
                    const cpu_array<Vectorr> &_locations = {},
                    const cpu_array<Vectorr> &_rotations = {});

  /// @brief Set the output formats
  /// @param _output_formats output formats
  void set_output_formats(const std::vector<std::string> &_output_formats);

  /// @brief allocates memory for rigid body level set
  /// @param nodes_ref Nodes container
  /// @param particles_ref Particles container
  void initialize(const NodesContainer &nodes_ref,
                  const ParticlesContainer &particles_ref) override;

  /// @brief calculates grid normals of rigid body level set
  /// @param nodes_ref Nodes container
  /// @param particles_ref Particles container
  void calculate_grid_normals(NodesContainer &nodes_ref,
                              ParticlesContainer &particles_ref);

  /// @brief finds the closest rigid particle to each grid node
  /// @param nodes_ref Nodes container
  /// @param particles_ref Particles container
  void calculate_overlapping_rigidbody(NodesContainer &nodes_ref,
                                       ParticlesContainer &particles_ref);

  /// @brief set velocities of rigid particles
  /// @param particles_ref Particles container
  void set_velocities(ParticlesContainer &particles_ref);

  /// @brief set position of rigid particles
  /// @param particles_ref Particles container
  void set_position(ParticlesContainer &particles_ref);

  /// @brief apply rigid body contact on background grid
  /// @param nodes_ref Nodes container
  /// @param particles_ref Particles container
  void apply_on_nodes_moments(NodesContainer &nodes_ref,
                              ParticlesContainer &particles_ref) override;

  void output_vtk(NodesContainer &nodes_ref,
                  ParticlesContainer &particles_ref) override;

  void setModeLoopRotate(Vectorr euler_angles_per_second, Real rate = 1.0);

  int mode = 0;

  bool is_animated;

  /// @brief Number of animation frames for rigid body
  int num_frames;

  /// @brief Rigid body center of mass of previous step
  Vectorr COM;

  /// @brief Rigid body euler angles of previous step
  Vectorr ROT;

  /// @brief Translational velocity of rigid body
  Vectorr translational_velocity;

  /// @brief Rotation matrix of rigid body
  Matrixr rotation_matrix;

  /// @brief Normals of non-rigid material points
  gpu_array<Vectorr> normals_gpu;

  /// @brief Flag if rigid grid node overlaps with non-rigid material point
  gpu_array<bool> is_overlapping_gpu;

  /// @brief Closest rigid particle to a node
  gpu_array<int> closest_rigid_particle_gpu;

  /// @brief Animations frames (steps)
  cpu_array<int> frames_cpu;

  /// @brief Animations locations
  cpu_array<Vectorr> locations_cpu;

  //// @brief Animations euler angles
  cpu_array<Vectorr> rotations_cpu;

  /// @brief Current animation frame
  int current_frame = 0;

  /// @brief Output formats
  /// @details supported formats are: "vtk" "obj"  "csv"
  std::vector<std::string> output_formats;

  /// @brief current euler angles of rigid body
  Vectorr euler_angles;

  /// @brief current angular velocity of rigid body
  Vectorr angular_velocities;
};
} // namespace pyroclastmpm