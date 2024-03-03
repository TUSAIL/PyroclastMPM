#include "pybind11/eigen.h"
#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/materials/dp_rheo.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pyroclastmpm {

void dp_rheo_module(const py::module &m) {

  py::class_<DP_rheo> material_cls(m, "DP_rheo",
                                            py::dynamic_attr());
  material_cls.def(py::init<Real, Real, Real, Real, Real, Real, Real, Real,
                            Real, Real, Real, Real>(),
                   R"(
                Drucker-Prager 
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
                dilatancy_angle: float
                    Dilatancy angle (degrees)
                    )",
                   py::arg("density"), py::arg("E"), py::arg("pois"),
                   py::arg("M"), py::arg("lam"), py::arg("kap"), py::arg("Vs"),
                   py::arg("R"), py::arg("Pt"), py::arg("beta"),
                   py::arg("dilatancy_angle") = (Real)0.0,
                   py::arg("solid_volume_fraction") = (Real)1.0

  );

  material_cls.def(
      "stress_update",
      [](DP_rheo &self, ParticlesContainer particles_ref, int mat_id) {
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
      [](DP_rheo &self, ParticlesContainer particles_ref, int mat_id) {
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
      [](DP_rheo &self) {
        return std::vector<Matrixr>(self.eps_e_gpu.begin(),
                                    self.eps_e_gpu.end());
      },
      [](DP_rheo &self, const std::vector<Matrixr> &value) {
        cpu_array<Matrixr> host_val = value;
        self.eps_e_gpu = host_val;
      },
      "Elastic strain (infinitesimal)");
  material_cls.def_property(
      "pc_gpu",
      [](DP_rheo &self) {
        return std::vector<Real>(self.pc_gpu.begin(), self.pc_gpu.end());
      },
      [](DP_rheo &self, const std::vector<Real> &value) {
        cpu_array<Real> host_val = value;
        self.pc_gpu = host_val;
      },
      "Preconsolidation pressure (updated)");
  material_cls.def_property(
      "solid_volume_fraction",
      [](DP_rheo &self) {
        return std::vector<Real>(self.solid_volume_fraction_gpu.begin(),
                                 self.solid_volume_fraction_gpu.end());
      },
      [](DP_rheo &self, const std::vector<Real> &value) {
        cpu_array<Real> host_val = value;
        self.solid_volume_fraction_gpu = host_val;
      },
        "Solid volume fraction");
  material_cls.def_readwrite("beta", &DP_rheo::beta,
                             "Size of outer eclipse");
  material_cls.def_readwrite("M", &DP_rheo::M, "CSL slope");
  material_cls.def_readwrite("Pt", &DP_rheo::Pt, "Tensile pressure");
  material_cls.def_readwrite("E", &DP_rheo::E, "Young's modulus");
  material_cls.def_readwrite("pois", &DP_rheo::pois,
                             "Poisson's ratio");
  material_cls.def_readwrite("shear_modulus", &DP_rheo::shear_modulus,
                             "Shear modulus G");
  material_cls.def_readwrite("lame_modulus", &DP_rheo::lame_modulus,
                             "Lame modulus lambda");
  material_cls.def_readwrite("bulk_modulus", &DP_rheo::bulk_modulus,
                             "Bulk modulus K");
  material_cls.def_readwrite("density", &DP_rheo::density,
                             "Bulk density of the material");
  material_cls.def_readwrite("do_update_history",
                             &DP_rheo::do_update_history,
                             "Flag if we update the history or not");
  material_cls.def_readwrite(
      "is_velgrad_strain_increment",
      &DP_rheo::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                               udpdate)");
  material_cls.def("calculate_timestep", &DP_rheo::calculate_timestep,
                   "calculate_timestep");
}

} // namespace pyroclastmpm