#include "pyroclastmpm/materials/muijop.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void mu_i_module(const py::module &m) {
  py::class_<MuIJop> material_cls(m, "MuIJop", py::dynamic_attr());
  material_cls.def(
      "stress_update",
      [](MuIJop &self, ParticlesContainer particles_ref, int mat_id) {
        self.stress_update(particles_ref, mat_id);
        return std::make_tuple(particles_ref, mat_id);
      },
      R"( ... docs missing)");

  material_cls.def(py::init<Real, Real, Real, Real>(),
              R"(
             mu I rheology of Jop

            Implementation is based on the paper:

            Jop, Pierre, Yoël Forterre, and Olivier Pouliquen. 
            "A constitutive law for dense granular flows."
            Nature 441.7094 (2006): 727-730.

             

             Example usage:
                >>> import pyroclastmpm as pm
                >>> mat = pm.MuIJop(1000, 1e-3, 1e6, 7)

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
  material_cls.def_readwrite("density", &MuIJop::density,
                        "Bulk density of the material");
  material_cls.def_readwrite("viscosity", &MuIJop::viscosity, "Viscosity");
  material_cls.def_readwrite("bulk_modulus", &MuIJop::bulk_modulus,
                        "Bulk modulus K");
  material_cls.def_readwrite("gamma", &MuIJop::gamma, "7 water and 1.4 for air");
  material_cls.def(py::pickle(
                  [](const MuIJop &a) {
                    return py::make_tuple(a.density, a.viscosity,
                                          a.bulk_modulus, a.gamma);
                  },
                  [](py::tuple t) {
                    auto mat = MuIJop{t[0].cast<Real>(), t[1].cast<Real>(),
                                      t[2].cast<Real>(), t[3].cast<Real>()};
                    return mat;
                  }),
              "Pickling for mu I rheology Jop");
  material_cls.def(
      "initialize",
      [](MuIJop &self, ParticlesContainer particles_ref, int mat_id) {
        self.initialize(particles_ref, mat_id);
        return std::make_tuple(particles_ref, mat_id);
      },
      R"(
              Initialize history variables

              Parameters
              ----------
              particles : ParticlesContainer
                  Particle container
              mat_id : int, optional
                  Material ID or colour, by default 0

              Returns
              -------
              Type[ParticlesContainer]
                  Particle container (initialized)
              )");


}

}