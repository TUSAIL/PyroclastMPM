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

#include "pyroclastmpm/common/types_common.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#if DIM == 2
#define PYBIND_MODULE_NAME MPM2D
#elif DIM == 1
#define PYBIND_MODULE_NAME MPM1D
#elif DIM == 3
#define PYBIND_MODULE_NAME MPM3D
#else
#error "Unsupported value for DDIM"
#endif

namespace pyroclastmpm {

void particles_module(const py::module &);
void nodes_module(const py::module &);
void boundaryconditions_module(const py::module &);
void materials_module(const py::module &);
void solver_module(const py::module &);
void global_settings_module(py::module &);
void tools_module(py::module &);

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {

  m.attr("global_dimension") = DIM;
  particles_module(m);
  nodes_module(m);
  boundaryconditions_module(m);
  materials_module(m);
  solver_module(m);

  global_settings_module(m);
  tools_module(m);
}

} // namespace pyroclastmpm