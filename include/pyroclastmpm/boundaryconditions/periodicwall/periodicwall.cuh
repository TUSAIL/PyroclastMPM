#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
#include "pyroclastmpm/boundaryconditions/periodicwall/periodicwall_kernels.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/particles/particles.cuh"

namespace pyroclastmpm {
/**
 * @brief Periodic walls. Applies periodic boundary conditions on particles and
 * nodes (at the wall).
 *
 */
struct PeriodicWall : BoundaryCondition {

  // FUNCTIONS

  /**
   * @brief Construct a new Periodic Wall object
   *
   * @param wallplane string containing an axis (x0,x1,y0,y1,z0,z1) where 0 is
   * at origin and 1 is at end
   */
  PeriodicWall(const std::string wallplane);
  ~PeriodicWall(){};

  /**
   * @brief Apply boundary conditions on node moments and masses
   *
   * @param nodes_ptr NodesContainer object
   */
  void apply_on_nodes_moments(NodesContainer& nodes_ref, ParticlesContainer & particles_ref) override;

  /**
   * @brief Apply boundary conditions on particles
   *
   * @param nodes_ptr NodesContainer object
   */
  void apply_on_particles(ParticlesContainer& particles_ptr) override;

  // VARIABLES

  /** @brief Key representing the axis `0 - x, 1 - y, 2 - z ` */
  int axis_key;

  /** @brief The barrier where the particle is transported to other plane */
  Real particle_barrier_cpu;
};

}  // namespace pyroclastmpm