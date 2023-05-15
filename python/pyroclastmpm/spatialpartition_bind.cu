#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/spatialpartition/spatialpartition.cuh"

namespace py = pybind11;


namespace pyroclastmpm {

void spatial_module(const py::module& m) {
  py::class_<SpatialPartition>(
      m, "SpatialPartition")
      .def(py::init<Vectorr, Vectorr, Real, int>(), py::arg("grid_start"),
           py::arg("grid_end"), py::arg("cell_size"), py::arg("num_elements"))
      .def_readonly("grid_start", &SpatialPartition::grid_start)
      .def_readonly("grid_end", &SpatialPartition::grid_end)
      .def_readonly("cell_size", &SpatialPartition::cell_size)
      .def_readonly("num_cells_total", &SpatialPartition::num_cells_total)
      .def_readonly("num_elements", &SpatialPartition::num_elements)
      .def_readonly("num_cells", &SpatialPartition::num_cells)
      // QUESTION Make these read only?
      .def_property(
          "cell_start",
          [](SpatialPartition& self) {
            return std::vector<int>(self.cell_start_gpu.begin(),
                                    self.cell_start_gpu.end());
          },  // getter
          [](SpatialPartition& self, const std::vector<int>& value) {
            cpu_array<int> host_val = value;
            self.cell_start_gpu = host_val;
          }  // setter
          )
      .def_property(
          "cell_end",
          [](SpatialPartition& self) {
            return std::vector<int>(self.cell_end_gpu.begin(),
                                    self.cell_end_gpu.end());
          },  // getter
          [](SpatialPartition& self, const std::vector<int>& value) {
            cpu_array<int> host_val = value;
            self.cell_end_gpu = host_val;
          }  // setter
          )
      .def_property(
          "sorted_index",
          [](SpatialPartition& self) {
            return std::vector<int>(self.sorted_index_gpu.begin(),
                                    self.sorted_index_gpu.end());
          },  // getter
          [](SpatialPartition& self, const std::vector<int>& value) {
            cpu_array<int> host_val = value;
            self.sorted_index_gpu = host_val;
          }  // setter
          )
      .def_property(
          "hash_unsorted",
          [](SpatialPartition& self) {
            return std::vector<unsigned int>(self.hash_unsorted_gpu.begin(),
                                             self.hash_unsorted_gpu.end());
          },  // getter
          [](SpatialPartition& self, const std::vector<unsigned int>& value) {
            cpu_array<unsigned int> host_val = value;
            self.hash_unsorted_gpu = value;
          }  // setter
          )
      .def_property(
          "hash_sorted",
          [](SpatialPartition& self) {
            return std::vector<unsigned int>(self.hash_sorted_gpu.begin(),
                                             self.hash_sorted_gpu.end());
          },  // getter
          [](SpatialPartition& self, const std::vector<unsigned int>& value) {
            cpu_array<unsigned int> host_val = value;
            self.hash_sorted_gpu = host_val;
          }  // setter
          )
      .def_property(
          "bins",
          [](SpatialPartition& self) {
            return std::vector<Vectori>(self.bins_gpu.begin(),
                                         self.bins_gpu.end());
          },  // getter
          [](SpatialPartition& self, const std::vector<Vectori>& value) {
            cpu_array<Vectori> host_val = value;
            self.bins_gpu = host_val;
          }  // setter
      );
}

}  // namespace pyroclastmpm