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

#include "pyroclastmpm/boundaryconditions/gravity.h"
#include "pyroclastmpm/common/types_common.h"

namespace py = pybind11;

namespace pyroclastmpm {

void gravity_module(const py::module &m) {
 
  py::class_<Gravity> boundarycondition_cls(m, "Gravity", py::dynamic_attr());
  boundarycondition_cls.def(py::init<Vectorr, bool, int, Vectorr>(),
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

  boundarycondition_cls.def_readwrite("gravity", &Gravity::gravity, "Gravity vector.");

};

} // namespace pyroclastmpm