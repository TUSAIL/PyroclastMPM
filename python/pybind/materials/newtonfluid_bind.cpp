#include "pyroclastmpm/materials/newtonfluid.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void newtonfluid_module(const py::module &m) {
 
  /* Newton Fluid */
  py::class_<NewtonFluid> material_cls(m, "NewtonFluid", py::dynamic_attr());
  material_cls.def(py::init<Real, Real, Real, Real>(),
             R"(
             Newtonian fluid

             Implementation is based on the paper:

             de Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory,
             implementation, and applications." Advances in applied mechanics 53 (2020):
             185-398. (Page 80)
             

             Example usage:
                >>> import pyroclastmpm as pm
                >>> mat = pm.NewtonFluid(1000, 1e-3, 1e6, 7)

              Parameters
              ----------
              density : float
                  Material density
              viscosity : float
                  Viscosity
              bulk_modulus : float, optional
                  Bulk modulus, by default 0.0
              gamma : float, optional
                  gamma used in EOS (7 water and 1.4 air), by default 7.0

             )",
             py::arg("density"), py::arg("viscosity"),
             py::arg("bulk_modulus") = 0., py::arg("gamma") = 7.);
  material_cls.def_readwrite("density", &NewtonFluid::density,
                       "Bulk density of the material");
  material_cls.def_readwrite("viscosity", &NewtonFluid::viscosity, "Viscosity");
  material_cls.def_readwrite("bulk_modulus", &NewtonFluid::bulk_modulus,
                       "Bulk modulus K");
  material_cls.def_readwrite("gamma", &NewtonFluid::gamma, "7 water and 1.4 for air");
  material_cls.def(py::pickle(
                 [](const NewtonFluid &a) {
                   return py::make_tuple(a.density, a.viscosity, a.bulk_modulus,
                                         a.gamma);
                 },
                 [](py::tuple t) {
                   auto mat = NewtonFluid{t[0].cast<Real>(), t[1].cast<Real>(),
                                          t[2].cast<Real>(), t[3].cast<Real>()};
                   return mat;
                 }),
             "Pickling for NewtonFluid");
}

}