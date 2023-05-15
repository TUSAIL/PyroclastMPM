// PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

// MODULE
#include "pyroclastmpm/common/tools.cuh"
#include "pyroclastmpm/common/types_common.cuh"

namespace py = pybind11;

namespace pyroclastmpm {

void tools_module(py::module& m) {
  m.def("uniform_random_points_in_volume", &uniform_random_points_in_volume,
        "Set the global dimension of the simulation");

  m.def("grid_points_in_volume", &grid_points_in_volume,
        "Set the global dimension of the simulation");

  m.def("grid_points_on_surface", &grid_points_on_surface,
        "Set the global dimension of the simulation");

  m.def("get_bounds", &get_bounds,
        "Set the global dimension of the simulation");


  m.def("set_device", &set_device,
        "Set the global dimension of the simulation");





//----------------------------------
  // void uniform_random_points_in_volume(const thrust::host_vector<Vector3r>
  // input,
  //                            const std::string stl_filename,
  //                            const int num_points)

  //   m.def("set_global_dimension", &set_global_dimension,
  //         "Set the global dimension of the simulation");
  //   m.def("set_global_shapefunction", &set_global_shapefunction,
  //         "Set the global shape function based on a dimension");
  //   m.def("set_global_timestep", &set_global_dt, "Set the global timestep");
  //   m.def("set_global_output_directory", &set_global_output_directory,
  //         "Set the global output directory");

  //   m.def(
  //       "set_globals", &set_globals,
  //       "Set the global dimension, shapefunction, timestep and output
  //       directory");
}

}  // namespace pyroclastmpm