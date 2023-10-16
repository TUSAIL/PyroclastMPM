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

// pybind
#include "pybind11/eigen.h"
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyroclastmpm/common/types_common.h"

#include "pyroclastmpm/solver/usl/usl.h"

namespace py = pybind11;

namespace pyroclastmpm {

extern const Real dt_cpu;

extern const int global_step_cpu;

void usl_module(const py::module &m) {

  py::class_<USL, Solver>(m, "USL").def(
      py::init<ParticlesContainer, NodesContainer, std::vector<MaterialType>,
               std::vector<BoundaryConditionType>, Real>(),
      R"(
      Update Stress Last solver. The implementation is based on the paper:
      
      de Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory,
      implementation, and applications." Advances in applied mechanics 53 (2020):
      185-398. (Page 32)

      Example usage:
          >>> import pyroclastmpm as pm
          >>> particles = pm.ParticlesContainer( ... )
          >>> nodes = pm.NodesContainer( ... )
          >>> materials = [pm.LinearElastic( ... )]
          >>> boundaryconditions = [pm.PlanarDomain( ... )]
          >>> solver = pm.USL(particles,nodes,materials,boundaryconditions)

      Parameters
      ----------
      particles : ParticlesContainer
          Particles storing the material points
      nodes : NodesContainer
          Nodes storing the background grid
      materials : [Material], optional
          List of constitutive models, by default None
      boundaryconditions : [BoundaryCondition], optional
          List of boundary conditions, by default None
      alpha : float, optional
            Flip/PIC mixture ratio, by default 0.99
      )",
      py::arg("particles"), py::arg("nodes"),
      py::arg("materials") = std::vector<MaterialType>(),
      py::arg("boundaryconditions") = std::vector<BoundaryConditionType>(),
      py::arg("alpha") = 0.99); // INIT
}
} // namespace pyroclastmpm