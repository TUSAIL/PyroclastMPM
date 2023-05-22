#pragma once

#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"

namespace pyroclastmpm
{

  struct BodyForce : BoundaryCondition
  {

    // FUNCTIONS
    BodyForce(const std::string _mode,
              const cpu_array<Vectorr> _values,
              const cpu_array<bool> _mask);

    ~BodyForce(){};

    void apply_on_nodes_f_ext(NodesContainer &nodes_ptr) override;
    void apply_on_nodes_moments(NodesContainer &nodes_ref, ParticlesContainer &particles_ref) override;

    // VARIABLES

    gpu_array<Vectorr> values_gpu;

    gpu_array<bool> mask_gpu;

    int mode_id; // 0 forces, 1 is moments, 2 fixed moments
  };

}