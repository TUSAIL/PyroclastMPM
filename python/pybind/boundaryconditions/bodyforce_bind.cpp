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
#include "pyroclastmpm/common/types_common.h"

namespace py = pybind11;

namespace pyroclastmpm {

void bodyforce_module(const py::module &m) {

  /*BodyForce*/
  py::class_<BodyForce> boundarycondition_cls(m, "BodyForce", py::dynamic_attr());
  boundarycondition_cls.def(py::init<std::string, std::vector<Vectorr>, std::vector<bool>>(),
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
};

} // namespace pyroclastmpm