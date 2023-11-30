#include "pyroclastmpm/materials/modifiedcamclay_nl.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void modified_camclay_nl_module(const py::module &m) {

 py::class_<ModifiedCamClayNonLinear> material_cls(m, "ModifiedCamClayNonLinear", py::dynamic_attr());
  material_cls.def(
      py::init<Real, Real, Real, Real, Real, Real, Real, Real, Real>(),
      R"(
                Non Linear Modified Cam Clay
                (infinitesimal strain)
                i.e. Modified Cam Clay with pressure dependent bulk modulus.

                Implementation based on the book:
                de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen.
                Computational methods for plasticity: theory and applications.
                John Wiley & Sons, 2011.

                Parameters
                ----------
                density : float
                    Material density
                pois : float
                    Poisson's ratio
                M : float
                    slope of the critical state line
                lam: float
                    slope of the virgin consolidation line
                kap: float
                    slope of the swelling line
                Vs: float
                    solid volume
                Pt : float
                    Tensile yield hydrostatic stress
                beta: float
                    Parameter related to size of outer diameter of ellipse
                    )",
      py::arg("density"),py::arg("pois"), py::arg("M"),
      py::arg("lam"), py::arg("kap"), py::arg("Vs"), py::arg("R"),
      py::arg("Pt"), py::arg("beta"));

  material_cls.def(
      "stress_update",
      [](ModifiedCamClayNonLinear &self, ParticlesContainer particles_ref, int mat_id) {
        self.stress_update(particles_ref, mat_id);
        return std::make_tuple(particles_ref, mat_id);
      },
      R"(
                Perform a stress update step.

                Returns
                -------
                ParticlesContainer
                    Particle container (updated stress)
              )");
  material_cls.def(
      "initialize",
      [](ModifiedCamClayNonLinear &self, ParticlesContainer particles_ref, int mat_id) {
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
  material_cls.def_property(
      "eps_e",
      [](ModifiedCamClayNonLinear &self) {
        return std::vector<Matrixr>(self.eps_e_gpu.begin(),
                                    self.eps_e_gpu.end());
      },
      [](ModifiedCamClayNonLinear &self, const std::vector<Matrixr> &value) {
        cpu_array<Matrixr> host_val = value;
        self.eps_e_gpu = host_val;
      },
      "Elastic strain (infinitesimal)");
  material_cls.def_property(
      "pc_gpu",
      [](ModifiedCamClayNonLinear &self) {
        return std::vector<Real>(self.pc_gpu.begin(), self.pc_gpu.end());
      },
      [](ModifiedCamClayNonLinear &self, const std::vector<Real> &value) {
        cpu_array<Real> host_val = value;
        self.pc_gpu = host_val;
      },
      "Preconsolidation pressure (updated)");
  material_cls.def_readwrite("E", &ModifiedCamClayNonLinear::E, "Young's modulus");
  material_cls.def_readwrite("pois", &ModifiedCamClayNonLinear::pois, "Poisson's ratio");
  material_cls.def_readwrite("shear_modulus", &ModifiedCamClayNonLinear::shear_modulus,
                        "Shear modulus G");
  material_cls.def_readwrite("lame_modulus", &ModifiedCamClayNonLinear::lame_modulus,
                        "Lame modulus lambda");
  material_cls.def_readwrite("bulk_modulus", &ModifiedCamClayNonLinear::bulk_modulus,
                        "Bulk modulus K");
  material_cls.def_readwrite("density", &ModifiedCamClayNonLinear::density,
                        "Bulk density of the material");
  material_cls.def_readwrite("do_update_history",
                        &ModifiedCamClayNonLinear::do_update_history,
                        "Flag if we update the history or not");
  material_cls.def_readwrite(
      "is_velgrad_strain_increment",
      &ModifiedCamClayNonLinear::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                               udpdate)");
  material_cls.def("calculate_timestep", &ModifiedCamClayNonLinear::calculate_timestep,
              "calculate_timestep");



}

}