#pragma once

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/particles/particles.cuh"

namespace pyroclastmpm {

/*!
 * @brief Boundary condition base class. The functions are called from a Solver class.
 */
struct BoundaryCondition {

  // FUNCTIONS

  /*!
   * @brief default constructor
   */
  BoundaryCondition() = default;

  /*!
   * @brief default destructor
   */
  ~BoundaryCondition() = default;

  /*!
   * @brief Apply on node forces
   * @param nodes_ref refernce to node container
   */
  virtual void apply_on_nodes_loads(NodesContainer& nodes_ref){};

  /*!
   * @brief Apply on node moments
   * @param nodes_ref reference to node container
   */
  virtual void apply_on_nodes_moments(NodesContainer& nodes_ref, ParticlesContainer & particles_ref){};

  /*!
   * @brief Apply on nodes forces
   * @param nodes_ref reference to node container
   */
  virtual void apply_on_nodes_f_ext(NodesContainer& nodes_ref){};

  /*!
   * @brief Apply on particles
   * @param particles_ref reference to particles
   */
  virtual void apply_on_particles(ParticlesContainer& particles_ref){};

  /*!
   * @brief Output after certain number of steps 
   */
  virtual void output_vtk(){};

  // VARIABLES

  /*!
   * @brief Enum of type of boundary condition (NodeBoundaryCondition,
   * ParticleBoundaryCondition)
   */
  BCType type;


  /**
   * @brief Is the boundary condition active or not
   * 
   */
  bool isActive= true;
};

}  // namespace pyroclastmpm