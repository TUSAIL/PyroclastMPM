#pragma once

#include "../common/types_common.h"
#include "pyroclastmpm/nodes/nodes.h"
#include "pyroclastmpm/particles/particles.h"

namespace pyroclastmpm {

void calculate_shape_function(NodesContainer &nodes_ref,
                              ParticlesContainer &particles_ref);

}
