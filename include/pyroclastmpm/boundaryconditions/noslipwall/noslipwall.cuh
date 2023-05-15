#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
#include "pyroclastmpm/boundaryconditions/noslipwall/noslipwall_kernels.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"

namespace pyroclastmpm {

/**
 * @brief A apply a no slip wall to an axis.
 *
 */
struct NoSlipWall : BoundaryCondition {

  // FUNCTIONS
  /**
   * @brief Construct a new N Slip Wall object
   *
   * @param wallplane string containing an axis (x0,x1,y0,y1,z0,z1) where 0 is
   * at origin and 1 is at end
   */
  NoSlipWall(const std::string wallplane);


  ~NoSlipWall(){};
  /**p
   * @brief Apply boundary conditions on node moments
   *
   * @param nodes_ptr NodesContainer object
   */
  void apply_on_nodes_moments(NodesContainer& nodes_ref, ParticlesContainer & particles_ref) override;

  // VARIABLES

  /** @brief Key representing the axis `0 - x, 1 - y, 2 - z ` */
  int axis_key;

  /** @brief Key representing the plan `0 - origin, 1 - end` */
  int plane_key;
};

}  // namespace pyroclastmpm