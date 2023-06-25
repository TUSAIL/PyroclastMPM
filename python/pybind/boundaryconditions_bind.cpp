// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyroclastmpm/boundaryconditions/bodyforce.h"
#include "pyroclastmpm/boundaryconditions/boundaryconditions.h"
#include "pyroclastmpm/boundaryconditions/gravity.h"
#include "pyroclastmpm/boundaryconditions/nodedomain.h"
#include "pyroclastmpm/boundaryconditions/planardomain.h"
#include "pyroclastmpm/boundaryconditions/rigidbodylevelset.h"
#include "pyroclastmpm/common/types_common.h"

namespace py = pybind11;

namespace pyroclastmpm {

void boundaryconditions_module(const py::module &m) {
  /*BoundaryCondition*/
  py::class_<BoundaryCondition> BC_cls(m, "BoundaryCondition",
                                       py::dynamic_attr());
  BC_cls.def(py::init<>(), "Base class of the boundaryscondition.");

  /*BodyForce*/
  py::class_<BodyForce> BF_cls(m, "BodyForce", py::dynamic_attr());
  BF_cls.def(py::init<std::string, std::vector<Vectorr>, std::vector<bool>>(),
             R"(
            Applies a body force or moment to the background grid based
            on the specified mode.

            Mode "forces" means the body forces are applied on the external
            forces of the background grid nodes.

            Mode "moments" means the body forces are applied on the moments
            of the background grid nodes.

            Mode "fixed" means node moments are fixed to a value.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> import numpy as np
                  >>> values = np.array([[0, -9.81, 0],[0, -0.81, 0] ...])
                  >>> bodyforce_bc = pm.BodyForce("forces", values, mask)

            Parameters
            ----------
            mode: int
                Mode of the boundary condition "forces", "moments" or "fixed"
            value: np.array
                Values of the moments/forces. Should be the same shape as the background
                grid nodes.
            mask: np.array
                Mask of the nodes that the boundary condition is applied to. Should be the same length
                as the background grid nodes.

            Tip
            -----
            :py:meth:`~NodesContainer.give_coords` is used to coordinates of the nodes
            that can help to specify the mask.

            )",
             py::arg("mode"), py::arg("values"), py::arg("mask"));

  py::class_<Gravity> G_cls(m, "Gravity", py::dynamic_attr());

  G_cls.def(py::init<Vectorr, bool, int, Vectorr>(),
            R"(
            Applies gravity to the background grid nodes, either be ramped
            up linearly or be constant.

            Example usage (constant):
                  >>> import pyroclastmpm as pm
                  >>> import numpy as np
                  >>> gravity_bc = pm.Gravity(np.array([0.0, -9.81, 0.0]))

            Example usage (ramped):
                  >>> import pyroclastmpm as pm
                  >>> import numpy as np
                  >>> gravity = pyroclastmpm.Gravity(
                  >>>   np.array([0.0, 0, 0.0]), true,
                  >>>   1000, np.array([0.0, -9.81, 0.0]))

            Parameters
            ----------
            gravity : np.array
                Gravity vector.
            is_ramp : bool, optional
                Flag if gravity is ramping, by default False.
            ramp_step : int, optional
                Number of steps to ramp up gravity, by default 0.
            gravity_end : np.array, optional
                Gravity at the end of ramp, by default None.

            )",
            py::arg("gravity"), py::arg("is_ramp") = false,
            py::arg("ramp_step") = 0, py::arg("gravity_end") = Vectorr::Zero());

  G_cls.def_readwrite("gravity", &Gravity::gravity, "Gravity vector.");

  py::class_<RigidBodyLevelSet> RBL_cls(m, "RigidBodyLevelSet",
                                        py::dynamic_attr());
  RBL_cls.def(py::init<Vectorr, std::vector<int>, std::vector<Vectorr>,
                       std::vector<Vectorr>>(),
              R"(
            A rigid body consists of a set of (MPM)rigid particles that are connected. 
            Rigid particles belong in the same array as the particles, but are mask as
            `is_rigid` in :class:`ParticlesContainer`


            A motion .chan file can be parsed to input the animation frames.
            For more information on .chan files
            see https://docs.blender.org/manual/en/latest/addons/import_export/anim_nuke_chan.html .


            Example usage:

            .. highlight:: python
            .. code-block:: python

                  import pyroclastmpm as pm
                  import numpy as np

                  # make rigid paticles and material particles
                  rigid_body = np.array([...]) # shape R, D
                  material = np.array([...]) # shape N, D

                  # combine particles and flag rigid particles
                  positions = np.concatenate((material, rigid_body), axis=0) # shape N+R, D
                  rigid_mask = np.array([...]) # shape N+R, 1 

                  # initialize ParticlesContainer
                  particles = pm.ParticlesContainer(positions, rigid_mask)

                  # initialize RigidBodyLevelSet boundary condition
                  rigidbody_bc = pm.RigidBodyLevelSet()
                  rigidbody_bc.set_output_formats(["vtk","csv"])

            Parameters
            ----------
            COM : np.ndarray, optional
                Center of mass of rigid body, required only if animated, by default None.
            frames : np.ndarray, optional
                Animation frames, by default None.
            locations : np.ndarray, optional
                Animation locations, by default None.
            rotations : np.ndarray, optional
                Animation rotations, by default None.
            output_formats : List[str], optional
                List of output formats, by default None.

            Tip
            -----
            The function :func:`grid_points_on_surface` is helpful convert STL files to
            rigid particles.

            )",
              py::arg("COM") = Vectorr::Zero(),
              py::arg("frames") = std::vector<int>(),
              py::arg("locations") = std::vector<Vectorr>(),
              py::arg("rotations") = std::vector<Vectorr>());

  RBL_cls.def("set_output_formats", &RigidBodyLevelSet::set_output_formats,
              R"(
            Set a list of formats that are outputted in a
            directory specified in :func:`set_globals`.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> rigidbody_bc = pm.RigidBodyLevelSet(...)
                  >>> rigidbody_bc.set_output_formats(["vtk","csv"])

            Parameters
            ----------
            output_formats : List[str]
                List of output formats.

            )",
              py::arg("output_formats"));

  py::class_<PlanarDomain> PD_cls(m, "PlanarDomain", py::dynamic_attr());
  PD_cls.def(py::init<Vectorr, Vectorr>(),
             R"(
            DEM style boundary conditions between particles and walls at
            outer domain with friction.

            How to choose parameters:

            Walls are defined by faces face0 =(x0, y0, z0) and face1 =(x1, y1, z1),
            for example in 2D it is

            .. image:: example.png

            Friction can be applied to each wall by specifying the friction in
            face0_friction and face1_friction

            Example usage in 3D:
                >>> import pyroclastmpm as pm
                >>> # floor friction of 15 degrees, walls and roof no friction
                >>> planar_bc = pm.PlanarDomain(
                >>> [0,15,0], [0,0,0])


            Parameters
            -----------
            face1_friction : np.array, optional
                Friction angle (degrees) for cube face x0,y0,z0, by default None.
            face1_friction : np.array, optional
                Friction angle (degrees)for cube face x1,y1,z1, by default None.
    
    )",
             py::arg("face0_friction") = Vectorr::Zero(),
             py::arg("face1_friction") = Vectorr::Zero());

  py::class_<NodeDomain> ND_cls(m, "NodeDomain", py::dynamic_attr());
  ND_cls.def(py::init<Vectori, Vectori>(),

             R"(
            Walls at outer domain resolved on the background grid nodes.

            How to choose parameters:

            Walls are defined by faces face0 =(x0, y0, z0) and face1 =(x1, y1, z1),
            for example in 2D it is
            
            .. image:: example.png

            Modes can be applied to each wall by specifying the friction in
            face0_mode and face1_mode. mode 0 - roller, mode 1 - fixed

            Parameters
            -----------
            face0_mode : np.array
                Roller or fixed modes for outside domain.
            face1_mode : np.array, optional
                Roller or fixed modes for outside domain.
            )",
             py::arg("face0_mode") = Vectori::Zero(),
             py::arg("face1_mode") = Vectori::Zero());
};

} // namespace pyroclastmpm