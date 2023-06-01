#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.h"

// #include "pyroclastmpm/nodes/nodes.cuh"

namespace pyroclastmpm {

/**
 * @brief Apply a domain to the simulation
 *
 */
struct PlanarDomain : BoundaryCondition {

  // FUNCTIONS
  /**
   * @brief Construct a new Pinball domain object
   *
   * @param axis0_friction friction of the x0,y0,z0 axes
   * @param axis1_friction friction of the x1,y1,z1 axes
   */
  PlanarDomain(Vectorr _axis0_friction = Vectorr::Zero(),
               Vectorr _axis1_friction = Vectorr::Zero());

  ~PlanarDomain(){};

  Vectorr axis0_friction;

  Vectorr axis1_friction;
  /**
   * @brief Apply boundary conditions on node moments
   *
   * @param nodes_ptr NodesContainer object
   */
  void apply_on_particles(ParticlesContainer &particles_ref) override;
};

} // namespace pyroclastmpm