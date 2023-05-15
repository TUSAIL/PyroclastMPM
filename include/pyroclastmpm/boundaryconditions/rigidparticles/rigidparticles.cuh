#pragma once

#include <thrust/execution_policy.h>

#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
#include "pyroclastmpm/boundaryconditions/rigidparticles/rigidparticles_kernels.cuh"
#include "pyroclastmpm/common/output.cuh"
#include "pyroclastmpm/common/helper.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/spatialpartition/spatialpartition.cuh"

namespace pyroclastmpm
{

  /**
   * @brief Apply rigid particle boundary conditions
   *
   */
  struct RigidParticles : BoundaryCondition
  {
    // FUNCTIONS

    RigidParticles(const cpu_array<Vectorr> _positions,
                   const cpu_array<int> _frames = {},
                   const cpu_array<Vectorr> _locations = {},
                   const cpu_array<Vectorr> _rotations = {},
                   const cpu_array<OutputType> _output_formats = {}

    );
    ~RigidParticles(){};

    void initialize(NodesContainer &nodes_ref, ParticlesContainer &particles_ref);

    void calculate_non_rigid_grid_normals(NodesContainer &nodes_ref,
                                          ParticlesContainer &particles_ref);

    void calculate_overlapping_rigidbody(NodesContainer &nodes_ref,
                                         ParticlesContainer &particles_ref);

    void update_grid_moments(NodesContainer &nodes_ref,
                             ParticlesContainer &particles_ref);

    void update_rigid_body(NodesContainer &nodes_ref,
                           ParticlesContainer &particles_ref);

    void find_nearest_rigid_body(NodesContainer &nodes_ref,
                                 ParticlesContainer &particles_ref);

    void partition();

    void apply_on_nodes_moments(NodesContainer &nodes_ref,
                                ParticlesContainer &particles_ref) override;

    void output_vtk() override;

    void calculate_velocities();

    // VARIABLES
    /** @brief number of rigid material points*/
    int num_particles;

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

    /** @brief particles' coordinates */
    gpu_array<Vectorr> positions_gpu;

    gpu_array<Vectorr> velocities_gpu;

    GPULaunchConfig launch_config;

    /** @brief spatial partitioning class */
    SpatialPartition spatial;

    /** @brief animations frames (steps) */
    cpu_array<int> frames_cpu;

    /** @brief animations locations */
    cpu_array<Vectorr> locations_cpu;

    /** @brief animations euler angles */
    cpu_array<Vectorr> rotations_cpu;

    cpu_array<OutputType> output_formats;
  };
} // namespace pyroclastmpm
