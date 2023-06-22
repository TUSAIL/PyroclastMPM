C++ documentation
===================


MPM
-------------------------

Solver
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: pyroclastmpm::USL
    :project: pyroclastmpm
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::Solver
    :project: pyroclastmpm
    :members:
    :undoc-members:

Particles
^^^^^^^^^^^^^^^^^^
.. doxygenclass:: pyroclastmpm::ParticlesContainer
    :project: pyroclastmpm
    :members:
    :undoc-members:

Nodes
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: pyroclastmpm::NodesContainer
    :project: pyroclastmpm
    :members:
    :undoc-members:

Constitutive models
^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: pyroclastmpm::LinearElastic
    :project: pyroclastmpm
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::MohrCoulomb
    :project: pyroclastmpm
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::LocalGranularRheology
    :project: pyroclastmpm
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::VonMises
    :project: pyroclastmpm
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::NewtonFluid
    :project: pyroclastmpm
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::Material
    :project: pyroclastmpm
    :members:
    :undoc-members:

Shape functions
^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: calculate_shape_function

Boundary conditions
^^^^^^^^^^^^^^^^^^^^^
.. doxygenclass:: pyroclastmpm::Gravity
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::BodyForce
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::NodeDomain
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::PlanarDomain
    :members:
    :undoc-members:

.. doxygenclass:: pyroclastmpm::RigidBodyLevelSet
    :members:
    :undoc-members:



Helper components
-----------------------

Input
^^^^^^^

.. doxygenfunction:: uniform_random_points_in_volume

.. doxygenfunction:: grid_points_in_volume

.. doxygenfunction:: grid_points_on_surface

Output
^^^^^^^

.. doxygenfunction:: set_vtk_points

.. doxygenfunction:: set_vtk_pointdata

.. doxygenfunction:: write_vtk_polydata

Arrays
^^^^^^^
.. doxygenfunction:: set_default_device

.. doxygenfunction:: reorder_device_array

.. doxygenfunction:: print_array

Spatial Partition
^^^^^^^^^^^^^^^^^^
.. doxygenclass:: pyroclastmpm::SpatialPartition
    :project: pyroclastmpm
    :members:
    :undoc-members:


Common
-------------------

Types
^^^^^^^^^

.. doxygenfile:: types_common.h
    :project: pyroclastmpm
    :sections: briefdescription innerclass public-type

Global variables and functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: set_globals

.. doxygenfunction:: set_global_shapefunction

.. doxygenfunction:: set_global_dt

.. doxygenfunction:: set_global_output_directory

.. doxygenfunction:: set_global_particles_per_cell

.. doxygenfunction:: set_global_step