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

#include "pyroclastmpm/boundaryconditions/rigidbodylevelset.h"
#include "pyroclastmpm/common/types_common.h"

namespace py = pybind11;

namespace pyroclastmpm {

void rigidbodylevelset_module(const py::module &m) {
 
  py::class_<RigidBodyLevelSet> boundarycondition_cls(m, "RigidBodyLevelSet",
                                        py::dynamic_attr());
  boundarycondition_cls.def(py::init<Vectorr, std::vector<int>, std::vector<Vectorr>,
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
                  rigid_body = np.array([...])[0] # shape R, D
                  material = np.array([...]) # shape N, D

                  # combine particles and flag rigid particles
                  positions = np.concatenate((material, rigid_body), axis=0) # shape N+R, D
                  rigid_mask = np.array([...]) # shape N+R, 1 

                  # initialize ParticlesContainer
                  particles = pm.ParticlesContainer(positions, rigid_mask)

                  # initialize RigidBodyLevelSet boundary condition
                  rigidbody_bc = pm.RigidBodyLevelSet()
                  rigidbody_bc.set_output_formats(["vtk","csv"])
                  rigidbody_bc.initialize(nodes,particles)

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

  boundarycondition_cls.def("set_output_formats", &RigidBodyLevelSet::set_output_formats,
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

  boundarycondition_cls.def("initialize", &RigidBodyLevelSet::initialize);
};

} // namespace pyroclastmpm