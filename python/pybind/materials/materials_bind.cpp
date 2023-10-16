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
#include "pyroclastmpm/materials/materials.h"
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;

namespace pyroclastmpm {

/**
 * @brief Bindings for materials
 *
 * @param m pybind11 module
 */
void materials_module(const py::module &m) {

  /* Material Base */
  py::class_<Material> material_cls(m, "Material", py::dynamic_attr());
  material_cls.def(py::init<>(), "Material base class");
  material_cls.def_readwrite("density", &Material::density,
                      "Bulk density of the material");
  material_cls.def(
      py::pickle([](const Material &a) { return py::make_tuple(a.density); },
                 [](py::tuple t) { return Material{t[0].cast<Real>()}; }),
      "Material pickle");

};

} // namespace pyroclastmpm