
from .pyroclastmpm_pybind import global_dimension

from .particles import ParticlesContainer
from .nodes import NodesContainer

from .boundaryconditions import (
    Gravity,
    BoundaryCondition,
    BodyForce,
    RigidBodyLevelSet,
    PlanarDomain,
    NodeDomain
)

from .materials import (
    Material,
    LinearElastic,
    NewtonFluid,
    LocalGranularRheology,
    VonMises
)

from .solver import (
    USL,
    #  TLMPM,
    #  MUSL
)

from .pyroclastmpm_pybind import (
    LinearShapeFunction,
    CubicShapeFunction,
    #     QuadraticShapeFunction, // TODO: implement
)

from .pyroclastmpm_pybind import (
    VTK,
    CSV,
    OBJ,
)

from .global_settings import (
    set_global_shapefunction,
    set_globals,
    set_global_output_directory,
    set_global_timestep,
    set_global_step
)

from .tools import (
    uniform_random_points_in_volume,
    grid_points_in_volume,
    grid_points_on_surface,
    get_bounds,
    set_device,
)
