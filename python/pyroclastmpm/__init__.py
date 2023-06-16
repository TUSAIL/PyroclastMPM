# BSD 3-Clause License
# Copyright (c) 2023, Retief Lubbe
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this
#  list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# trunk-ignore-all(ruff/F401)
from .boundaryconditions import (
    BodyForce,
    BoundaryCondition,
    Gravity,
    NodeDomain,
    PlanarDomain,
    RigidBodyLevelSet,
)
from .global_settings import (
    set_global_output_directory,
    set_global_shapefunction,
    set_global_step,
    set_global_timestep,
    set_globals,
)
from .helper import check_dimension, global_dimension
from .materials import (
    LinearElastic,
    LocalGranularRheology,
    Material,
    MohrCoulomb,
    NewtonFluid,
    VonMises,
)
from .nodes import NodesContainer
from .particles import ParticlesContainer
from .solver import USL  # TLMPM,; MUSL
from .tools import (
    get_bounds,
    grid_points_in_volume,
    grid_points_on_surface,
    set_device,
    uniform_random_points_in_volume,
)
