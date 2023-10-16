#include "pyroclastmpm/materials/localrheo.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void localrheology_module(const py::module &m) {

 
  /* Local Rheology */
  py::class_<LocalGranularRheology> material_cls(m, "LocalGranularRheology",
                                            py::dynamic_attr());
  material_cls.def(py::init<Real, Real, Real, Real, Real, Real, Real, Real, Real>(),
              R"(
              Local granular rheology

              The implementation is based on the paper
              Dunatunga, Sachith, and Ken Kamrin.
              Continuum modelling and simulation of granular flows through their many phases.
              Journal of Fluid Mechanics 779 (2015): 483-513.

              (Warning unstable)

              Parameters
              ----------
              density : float
                  Material density
              E : float
                  Yoiung's modulus
              pois : float
                  Poisson's ratio
              I0 : float
                  Inertial number
              mu_s : float
                  Critical friction angle (max)
              mu_2 : float
                  Critical friction angle (min)
              rho_c : float
                  Critical dnesity
              particle_diameter : float
                  Particle diameter
              particle_density : float
                  Particle solid density
                )",
              py::arg("density"), py::arg("E"), py::arg("pois"), py::arg("I0"),
              py::arg("mu_s"), py::arg("mu_2"), py::arg("rho_c"),
              py::arg("particle_diameter"), py::arg("particle_density"));
  material_cls.def_readwrite("density", &LocalGranularRheology::density,
                        "Bulk density of the material");
  material_cls.def_readwrite("E", &LocalGranularRheology::E, "Young's modulus");
  material_cls.def_readwrite("pois", &LocalGranularRheology::pois,
                        "Poisson's ratio");
  material_cls.def_readwrite("shear_modulus", &LocalGranularRheology::shear_modulus,
                        "Shear modulus G");
  material_cls.def_readwrite("lame_modulus", &LocalGranularRheology::lame_modulus,
                        "Lame modulus lambda");
  material_cls.def_readwrite("bulk_modulus", &LocalGranularRheology::bulk_modulus,
                        "Bulk modulus K");
  material_cls.def_readwrite("mu_s", &LocalGranularRheology::mu_s,
                        "Critical friction angle (max)");
  material_cls.def_readwrite("mu_2", &LocalGranularRheology::mu_2,
                        "Critical friction angle (min)");
  material_cls.def_readwrite("I0", &LocalGranularRheology::I0, "Inertial number");
  material_cls.def_readwrite("rho_c", &LocalGranularRheology::rho_c,
                        "Critical density");
  material_cls.def_readwrite("EPS", &LocalGranularRheology::EPS, "EPS");
  material_cls.def_readwrite("particle_density",
                        &LocalGranularRheology::particle_density,
                        "Particle density");
  material_cls.def_readwrite("particle_diameter",
                        &LocalGranularRheology::particle_diameter,
                        "Particle diameter");
  material_cls.def(
      py::pickle(
          [](const LocalGranularRheology &a) { // dump
            return py::make_tuple(a.density, a.E, a.pois, a.I0, a.mu_s, a.mu_2,
                                  a.rho_c, a.particle_diameter,
                                  a.particle_density, a.shear_modulus,
                                  a.lame_modulus, a.bulk_modulus, a.EPS);
          },
          [](py::tuple t) { // load
            auto mat = LocalGranularRheology{
                t[0].cast<Real>(), t[1].cast<Real>(), t[2].cast<Real>(),
                t[3].cast<Real>(), t[4].cast<Real>(), t[5].cast<Real>(),
                t[6].cast<Real>(), t[7].cast<Real>(), t[8].cast<Real>()};
            mat.shear_modulus = t[9].cast<Real>();
            mat.lame_modulus = t[10].cast<Real>();
            mat.bulk_modulus = t[11].cast<Real>();
            mat.EPS = t[12].cast<Real>();
            return mat;
          }),
      "Pickling for LocalGranularRheology");

}

}