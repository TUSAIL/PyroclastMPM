#include "pyroclastmpm/materials/modifiedcamclay.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void modified_camclay_module(const py::module &m) {

 py::class_<ModifiedCamClay> material_cls(m, "ModifiedCamClay", py::dynamic_attr());
  material_cls.def(
      py::init<Real, Real, Real, Real, Real, Real, Real, Real, Real, Real>(),
      R"(
                Modified Cam Clay
                (infinitesimal strain)
                Implementation based on the book:
                de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen.
                Computational methods for plasticity: theory and applications.
                John Wiley & Sons, 2011.

                Parameters
                ----------
                density : float
                    Material density
                E : float
                    Young's modulus
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
      py::arg("density"), py::arg("E"), py::arg("pois"), py::arg("M"),
      py::arg("lam"), py::arg("kap"), py::arg("Vs"), py::arg("R"),
      py::arg("Pt"), py::arg("beta"));

  material_cls.def(
      "stress_update",
      [](ModifiedCamClay &self, ParticlesContainer particles_ref, int mat_id) {
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
      [](ModifiedCamClay &self, ParticlesContainer particles_ref, int mat_id) {
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
      [](ModifiedCamClay &self) {
        return std::vector<Matrixr>(self.eps_e_gpu.begin(),
                                    self.eps_e_gpu.end());
      },
      [](ModifiedCamClay &self, const std::vector<Matrixr> &value) {
        cpu_array<Matrixr> host_val = value;
        self.eps_e_gpu = host_val;
      },
      "Elastic strain (infinitesimal)");
  material_cls.def_property(
      "pc_gpu",
      [](ModifiedCamClay &self) {
        return std::vector<Real>(self.pc_gpu.begin(), self.pc_gpu.end());
      },
      [](ModifiedCamClay &self, const std::vector<Real> &value) {
        cpu_array<Real> host_val = value;
        self.pc_gpu = host_val;
      },
      "Preconsolidation pressure (updated)");
  material_cls.def_readwrite("E", &ModifiedCamClay::E, "Young's modulus");
  material_cls.def_readwrite("pois", &ModifiedCamClay::pois, "Poisson's ratio");
  material_cls.def_readwrite("shear_modulus", &ModifiedCamClay::shear_modulus,
                        "Shear modulus G");
  material_cls.def_readwrite("lame_modulus", &ModifiedCamClay::lame_modulus,
                        "Lame modulus lambda");
  material_cls.def_readwrite("bulk_modulus", &ModifiedCamClay::bulk_modulus,
                        "Bulk modulus K");
  material_cls.def_readwrite("density", &ModifiedCamClay::density,
                        "Bulk density of the material");
  material_cls.def_readwrite("do_update_history",
                        &ModifiedCamClay::do_update_history,
                        "Flag if we update the history or not");
  material_cls.def_readwrite(
      "is_velgrad_strain_increment",
      &ModifiedCamClay::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                               udpdate)");
  material_cls.def("calculate_timestep", &ModifiedCamClay::calculate_timestep,
              "calculate_timestep");



}

}