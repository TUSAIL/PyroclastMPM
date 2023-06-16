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
  py::class_<BoundaryCondition>(m, "BoundaryCondition").def(py::init<>());

  py::class_<BodyForce>(m, "BodyForce")
      .def(py::init<std::string, std::vector<Vectorr>, std::vector<bool>>(),
           py::arg("mode"), py::arg("values"), py::arg("mask"))
      .def_readwrite("mode_id", &BodyForce::mode_id);

  py::class_<Gravity>(m, "Gravity")
      .def(py::init<Vectorr, bool, int, Vectorr>(), py::arg("gravity"),
           py::arg("is_ramp"), py::arg("ramp_step"), py::arg("gravity_end"))
      .def_readwrite("gravity", &Gravity::gravity);

  py::class_<RigidBodyLevelSet>(m, "RigidBodyLevelSet")
      .def(py::init<Vectorr, std::vector<int>, std::vector<Vectorr>,
                    std::vector<Vectorr>>(),
           py::arg("COM") = Vectorr::Zero(),
           py::arg("frames") = std::vector<int>(),
           py::arg("locations") = std::vector<Vectorr>(),
           py::arg("rotations") = std::vector<Vectorr>())
      .def("set_output_formats", &RigidBodyLevelSet::set_output_formats);

  py::class_<PlanarDomain>(m, "PlanarDomain")
      .def(py::init<Vectorr, Vectorr>(),
           py::arg("axis0_friction") = Vectorr::Zero(),
           py::arg("axis1_friction") = Vectorr::Zero());

  py::class_<NodeDomain>(m, "NodeDomain")
      .def(py::init<Vectori, Vectori>(),
           py::arg("axis0_mode") = Vectori::Zero(),
           py::arg("axis1_mode") = Vectori::Zero());
};

} // namespace pyroclastmpm