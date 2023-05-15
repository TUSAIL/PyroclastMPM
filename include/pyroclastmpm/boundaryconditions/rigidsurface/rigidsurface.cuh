#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
#include "pyroclastmpm/boundaryconditions/rigidparticles/rigidparticles_kernels.cuh"
#include "pyroclastmpm/common/output.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/spatialpartition/spatialpartition.cuh"

#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

#include <vtkSTLReader.h>

#include <vtkPolyDataNormals.h>
#include <vtkPolyDataPointSampler.h>
#include <vtkPolyDataMapper.h>
#include "vtkNew.h"

#include <vtkDistancePolyDataFilter.h>

namespace pyroclastmpm {

/**
 * @brief Apply rigid particle boundary conditions
 *
 */
struct RigidSurface : BoundaryCondition {
  // FUNCTIONS

  RigidSurface(const std::string stl_filename,
               const Real min_dist,
               const Vector3r _body_velocity);
  ~RigidSurface(){};

  void partition();

  void apply_on_nodes_moments(NodesContainer& nodes_ref,
                              ParticlesContainer& particles_ref) override;

  void apply_scripted_motion();

  void output_vtk() override;

  // VARIABLES

  int num_particles;

  Vector3r body_velocity;

  thrust::device_vector<Vector3r> normals_gpu;

  thrust::device_vector<bool> is_overlapping_gpu;

  /** @brief particles' coordinates */
  thrust::device_vector<Vector3r> positions_gpu;

  /** @brief particles' velocities */
  thrust::device_vector<Vector3r> velocities_gpu;

  /** @brief GPU Kernel launch configuration, number of threads per block */
  dim3 launch_grid_config;

  /** @brief GPU Kernel launch configuration,  number of blocks */
  dim3 launch_block_config;

  /** @brief spatial partitioning class */
  std::shared_ptr<SpatialPartition> spatial_ptr;

  /** @brief particles' shape functions */
  thrust::device_vector<Real> psi_gpu;

  /** @brief particles' shape function gradients */
  thrust::device_vector<Vector3r> dpsi_gpu;


  vtkSmartPointer<vtkPolyData> geometry;
};
}  // namespace pyroclastmpm
