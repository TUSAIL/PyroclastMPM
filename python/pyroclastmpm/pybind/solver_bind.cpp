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

// SOLVERS
#include "pyroclastmpm/solver/solver.h"
#include "pyroclastmpm/solver/usl/usl.h"

namespace py = pybind11;

namespace pyroclastmpm {

extern const Real dt_cpu;

extern const int global_step_cpu;

void solver_module(py::module &m) {
  // SOLVER BASE
  py::class_<Solver>(m, "Solver")
      .def(py::init<ParticlesContainer, NodesContainer,
                    std::vector<MaterialType>,
                    std::vector<BoundaryConditionType>>(),
           py::arg("particles"), py::arg("nodes"),
           py::arg("materials") = std::vector<MaterialType>(),
           py::arg("boundaryconditions") =
               std::vector<BoundaryConditionType>()) // INIT
      .def("solve_nsteps", &Solver::solve_nsteps)
      .def_readwrite("nodes", &Solver::nodes)         // NODES
      .def_readwrite("particles", &Solver::particles) // PARTICLES
      .def_property("current_time",
                    [](Solver &self) { return dt_cpu * global_step_cpu; },
                    {}) // CURRENT TIME
      .def_property("current_step",
                    [](Solver &self) { return global_step_cpu; },
                    {}) // CURRENT TIME
      .def_property(
          "boundaryconditions",
          [](Solver &self) {
            return std::vector<BoundaryConditionType>(
                self.boundaryconditions.begin(), self.boundaryconditions.end());
          }, // getter
          [](Solver &self, const std::vector<BoundaryConditionType> &value) {
            self.boundaryconditions = value;
          } // setter
      );

  // USL SOLVER
  py::class_<USL, Solver>(m, "USL").def(
      py::init<ParticlesContainer, NodesContainer, std::vector<MaterialType>,
               std::vector<BoundaryConditionType>, Real>(),
      py::arg("particles"), py::arg("nodes"),
      py::arg("materials") = std::vector<MaterialType>(),
      py::arg("boundaryconditions") = std::vector<BoundaryConditionType>(),
      py::arg("alpha") = 0.99); // INIT

  // // MUSL SOLVER
  // py::class_<MUSL, USL>(m, "MUSL").def(
  //     py::init<ParticlesContainer, NodesContainer, std::vector<MaterialType>,
  //              std::vector<BoundaryConditionType>, Real>(),
  //     py::arg("particles"), py::arg("nodes"),
  //     py::arg("materials") = std::vector<MaterialType>(),
  //     py::arg("boundaryconditions") =
  //         std::vector<BoundaryConditionType>(),
  //     py::arg("alpha") = 0.99); // INIT

  // // TLMPM SOLVER
  // py::class_<TLMPM, MUSL>(m, "TLMPM").def(py::init<ParticlesContainer,
  // NodesContainer, std::vector<MaterialType>,
  // std::vector<BoundaryConditionType>, Real>(), py::arg("particles"),
  // py::arg("nodes"), py::arg("materials") = std::vector<MaterialType>(),
  // py::arg("boundaryconditions") = std::vector<BoundaryConditionType>(),
  // py::arg("alpha") = 0.99); // INIT

  // // APIC SOLVER
  // py::class_<APIC, Solver>(m, "APIC").def(
  //     py::init<ParticlesContainer, NodesContainer, std::vector<MaterialType>,
  //              std::vector<BoundaryConditionType>>(),
  //     py::arg("particles"), py::arg("nodes"), py::arg("materials"),
  //     py::arg("boundaryconditions") = std::vector<BoundaryConditionType>());
}
} // namespace pyroclastmpm