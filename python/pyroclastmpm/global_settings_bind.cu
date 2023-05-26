// PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

// MODULE
#include "pyroclastmpm/common/global_settings.cuh"
#include "pyroclastmpm/common/types_common.cuh"

namespace py = pybind11;

namespace pyroclastmpm {

void global_settings_module(py::module& m) {
  py::enum_<SFType>(m, "ShapeFunction")
      .value("LinearShapeFunction", LinearShapeFunction)
      .value("QuadraticShapeFunction", QuadraticShapeFunction)
      .value("CubicShapeFunction", CubicShapeFunction)
      .export_values();


  py::enum_<OutputType>(m, "OutputType")
      .value("VTK", VTK)
      .value("OBJ", OBJ)
      .value("CSV", CSV)
      .export_values();

  m.def("set_global_shapefunction", &set_global_shapefunction,
        "Set the global shape function based on a dimension");
  m.def("set_global_timestep", &set_global_dt, "Set the global timestep");
  m.def("set_global_step", &set_global_step, "Set the global step");
  m.def("set_global_output_directory", &set_global_output_directory,
        "Set the global output directory");

  m.def(
      "set_globals", &set_globals,
      "Set the global dimension, shapefunction, timestep and output directory");
}

}  // namespace pyroclastmpm