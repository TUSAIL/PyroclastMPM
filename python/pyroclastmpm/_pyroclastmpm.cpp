#include "pyroclastmpm/common/types_common.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pyroclastmpm {

void particles_module(py::module &);
void nodes_module(py::module &);
void boundaryconditions_module(py::module &);
void materials_module(py::module &);
void solver_module(py::module &);
void global_settings_module(py::module &);
void tools_module(py::module &);

PYBIND11_MODULE(pyroclastmpm_pybind, m) {

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