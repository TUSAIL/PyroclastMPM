#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"

namespace pyroclastmpm
{

  struct NodeDomain : BoundaryCondition
  {

    NodeDomain(Vectori _axis0_mode = Vectori::Zero(), Vectori _axis1_mode = Vectori::Zero());

    ~NodeDomain(){};

    void apply_on_nodes_moments(NodesContainer &nodes_ref, ParticlesContainer &particles_ref) override;



    Vectori axis0_mode;
    Vectori axis1_mode;
  };

} // namespace pyroclastmpm