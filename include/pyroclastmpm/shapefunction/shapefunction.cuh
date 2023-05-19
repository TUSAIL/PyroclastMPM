#pragma once

#include "../common/types_common.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/particles/particles.cuh"


namespace pyroclastmpm {


void calculate_shape_function(NodesContainer& nodes_ref, ParticlesContainer & particles_ref);
                                

}