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

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/nodes/nodes.h"

namespace py = pybind11;

namespace pyroclastmpm {

void nodes_module(const py::module &m) {
  py::class_<NodesContainer>(m, "NodesContainer")
      .def(py::init<Vectorr, Vectorr, Real>(), py::arg("node_start"),
           py::arg("node_end"), py::arg("node_spacing"))
      .def("set_output_formats", &NodesContainer::set_output_formats)
      .def("give_coords", &NodesContainer::give_node_coords_stl)
      .def_readonly("node_start", &NodesContainer::node_start)
      .def_readonly("node_end", &NodesContainer::node_end)
      .def_readonly("num_nodes", &NodesContainer::num_nodes)
      .def_readonly("node_spacing", &NodesContainer::node_spacing)
      .def_readonly("num_nodes_total", &NodesContainer::num_nodes_total)
      .def_property(
          "moments",
          [](NodesContainer &self) {
            return std::vector<Vectorr>(self.moments_gpu.begin(),
                                        self.moments_gpu.end());
          }, // getter
          [](NodesContainer &self, const std::vector<Vectorr> &value) {
            cpu_array<Vectorr> host_val = value;
            self.moments_gpu = host_val;
          } // setter
          )
      .def_property(
          "moments_nt",
          [](NodesContainer &self) {
            return std::vector<Vectorr>(self.moments_nt_gpu.begin(),
                                        self.moments_nt_gpu.end());
          }, // getter
          [](NodesContainer &self, const std::vector<Vectorr> &value) {
            cpu_array<Vectorr> host_val = value;
            self.moments_nt_gpu = host_val;
          } // setter
          )
      .def_property(
          "forces_external",
          [](NodesContainer &self) {
            return std::vector<Vectorr>(self.forces_external_gpu.begin(),
                                        self.forces_external_gpu.end());
          }, // getter
          [](NodesContainer &self, const std::vector<Vectorr> &value) {
            cpu_array<Vectorr> host_val = value;
            self.forces_external_gpu = host_val;
          } // setter
          )
      .def_property(
          "forces_internal",
          [](NodesContainer &self) {
            return std::vector<Vectorr>(self.forces_internal_gpu.begin(),
                                        self.forces_internal_gpu.end());
          }, // getter
          [](NodesContainer &self, const std::vector<Vectorr> &value) {
            cpu_array<Vectorr> host_val = value;
            self.forces_internal_gpu = host_val;
          } // setter
          )
      .def_property(
          "forces_total",
          [](NodesContainer &self) {
            return std::vector<Vectorr>(self.forces_total_gpu.begin(),
                                        self.forces_total_gpu.end());
          }, // getter
          [](NodesContainer &self, const std::vector<Vectorr> &value) {
            cpu_array<Vectorr> host_val = value;
            self.forces_total_gpu = host_val;
          } // setter
          )
      .def_property(
          "masses",
          [](NodesContainer &self) {
            return std::vector<Real>(self.masses_gpu.begin(),
                                     self.masses_gpu.end());
          }, // getter
          [](NodesContainer &self, const std::vector<Real> &value) {
            cpu_array<Real> host_val = value;
            self.masses_gpu = host_val;
          } // setter
      );
}

} // namespace pyroclastmpm