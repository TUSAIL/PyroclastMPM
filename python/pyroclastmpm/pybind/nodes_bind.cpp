#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/nodes/nodes.h"

namespace py = pybind11;

namespace pyroclastmpm {

void nodes_module(py::module &m) {
  py::class_<NodesContainer>(m, "NodesContainer")
      .def(py::init<Vectorr, Vectorr, Real, std::vector<OutputType>>(),
           py::arg("node_start"), py::arg("node_end"), py::arg("node_spacing"),
           py::arg("output_formats") = std::vector<OutputType>())
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