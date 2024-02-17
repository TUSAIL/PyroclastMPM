#include "pyroclastmpm/materials/vonmises.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void vonmises_module(const py::module &m) {

  py::class_<VonMises> material_cls(m, "VonMises", py::dynamic_attr());
  material_cls.def(py::init<Real, Real, Real, Real, Real>(),
             R"(
        Associative Von Mises plasticity with linear isotropic strain hardening.
        (infinitesimal strain)
        Implementation based on the book:
        de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen.
        Computational methods for plasticity: theory and applications.
        John Wiley & Sons, 2011.


        Example usage:
            >>> import pyroclastmpm as pm
            >>> mat = pm.VonMises(1000, 1e6, 0.25,1e3,1e3)
        
        Parameters
        ----------
        density : float
            Material density
        E : float
            Young's modulus
        pois : float, optional
            Poisson's ratio, by default 0
        yield_stress : float, optional
            Yield stress, by default 0
        H : float, optional
            Linear hardening modulus, by default 1

            )",
             py::arg("density"), py::arg("E"), py::arg("pois"),
             py::arg("yield_stress"), py::arg("H"));
  material_cls.def(
      "stress_update",
      [](VonMises &self, ParticlesContainer particles_ref, int mat_id) {
        self.stress_update(particles_ref, mat_id);
        return std::make_tuple(particles_ref, mat_id);
      },
      R"(
        Perform a stress update step.

        Example usage:
            >>> import pyroclastmpm as pm
            >>> mat = pm.VonMises(1000, 1e6, 0.25,1e3,1e3)
        
            >>> particles = pm.ParticlesContainer(np.array([0.,0.,0]))
            >>> particles = mat.initialize(particles, 0)
            >>> particles = mat.stress_update(particles, 0)
        
        Returns
        -------
        ParticlesContainer
            Particle container (updated stress)
      )");
  material_cls.def(
      "initialize",
      [](VonMises &self, ParticlesContainer particles_ref, int mat_id) {
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
      [](VonMises &self) {
        return std::vector<Matrixr>(self.eps_e_gpu.begin(),
                                    self.eps_e_gpu.end());
      },
      [](VonMises &self, const std::vector<Matrixr> &value) {
        cpu_array<Matrixr> host_val = value;
        self.eps_e_gpu = host_val;
      },
      "Elastic strain (infinitesimal)");
  material_cls.def_readwrite("E", &VonMises::E, "Young's modulus");
  material_cls.def_readwrite("pois", &VonMises::pois, "Poisson's ratio");
  material_cls.def_readwrite("shear_modulus", &VonMises::shear_modulus,
                       "Shear modulus G");
  material_cls.def_readwrite("lame_modulus", &VonMises::lame_modulus,
                       "Lame modulus lambda");
  material_cls.def_readwrite("bulk_modulus", &VonMises::bulk_modulus,
                       "Bulk modulus K");
  material_cls.def_readwrite("density", &VonMises::density,
                       "Bulk density of the material");
  material_cls.def_readwrite("do_update_history", &VonMises::do_update_history,
                       "Flag if we update the history or not");
  material_cls.def_readwrite(
      "is_velgrad_strain_increment", &VonMises::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                       udpdate)");


}

}