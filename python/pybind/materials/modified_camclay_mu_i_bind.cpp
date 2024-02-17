#include "pyroclastmpm/materials/mcc_mu_i.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void modified_camclay_mu_i_module(const py::module &m) {


  /* Modified Cam Clay */
  py::class_<MCCMuI> MCC_MU_I_cls(m, "MCCMuI", py::dynamic_attr());
  MCC_MU_I_cls.def(
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
      py::arg("lam"), py::arg("kap"), py::arg("Vs"), py::arg("Pc0"),
      py::arg("Pt"), py::arg("beta"));

  MCC_MU_I_cls.def(
      "stress_update",
      [](MCCMuI &self, ParticlesContainer particles_ref, int mat_id) {
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
  MCC_MU_I_cls.def(
      "initialize",
      [](MCCMuI &self, ParticlesContainer particles_ref, int mat_id) {
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
  MCC_MU_I_cls.def_property(
      "eps_e",
      [](MCCMuI &self) {
        return std::vector<Matrixr>(self.eps_e_gpu.begin(),
                                    self.eps_e_gpu.end());
      },
      [](MCCMuI &self, const std::vector<Matrixr> &value) {
        cpu_array<Matrixr> host_val = value;
        self.eps_e_gpu = host_val;
      },
      "Elastic strain (infinitesimal)");
  MCC_MU_I_cls.def_property(
      "pc_gpu",
      [](MCCMuI &self) {
        return std::vector<Real>(self.pc_gpu.begin(), self.pc_gpu.end());
      },
      [](MCCMuI &self, const std::vector<Real> &value) {
        cpu_array<Real> host_val = value;
        self.pc_gpu = host_val;
      },
      "Preconsolidation pressure (updated)");
  MCC_MU_I_cls.def_readwrite("E", &MCCMuI::E, "Young's modulus");
  MCC_MU_I_cls.def_readwrite("pois", &MCCMuI::pois, "Poisson's ratio");
  MCC_MU_I_cls.def_readwrite("shear_modulus", &MCCMuI::shear_modulus,
                             "Shear modulus G");
  MCC_MU_I_cls.def_readwrite("lame_modulus", &MCCMuI::lame_modulus,
                             "Lame modulus lambda");
  MCC_MU_I_cls.def_readwrite("bulk_modulus", &MCCMuI::bulk_modulus,
                             "Bulk modulus K");
  MCC_MU_I_cls.def_readwrite("density", &MCCMuI::density,
                             "Bulk density of the material");
  MCC_MU_I_cls.def_readwrite("do_update_history", &MCCMuI::do_update_history,
                             "Flag if we update the history or not");
  MCC_MU_I_cls.def_readwrite(
      "is_velgrad_strain_increment", &MCCMuI::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                               udpdate)");
  MCC_MU_I_cls.def("calculate_timestep", &MCCMuI::calculate_timestep,
                   "calculate_timestep");

}

}