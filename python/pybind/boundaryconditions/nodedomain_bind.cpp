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

#include "pyroclastmpm/boundaryconditions/nodedomain.h"
#include "pyroclastmpm/common/types_common.h"

namespace py = pybind11;

namespace pyroclastmpm {

void nodedomain_module(const py::module &m) {
 
  py::class_<NodeDomain> boundarycondition_cls(m, "NodeDomain", py::dynamic_attr());
  boundarycondition_cls.def(py::init<Vectori, Vectori>(),
             R"(
            Walls at outer domain resolved on the background grid nodes.

            How to choose parameters:

            Walls are defined by faces face0 =(x0, y0, z0) and face1 =(x1, y1, z1),
            for example in 2D it is
            
            .. image:: example.png

            Modes can be applied to each wall by specifying the friction in
            face0_mode and face1_mode. mode 0 is Fixed, mode 1 - roller
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