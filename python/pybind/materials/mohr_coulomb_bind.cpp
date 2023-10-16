#include "pyroclastmpm/materials/mohrcoulomb.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void mohr_coulomb_module(const py::module &m) {

  /*Mohr Coulomb*/
  py::class_<MohrCoulomb> material_cls(m, "MohrCoulomb");
  material_cls.def(py::init<Real, Real, Real, Real, Real, Real, Real>(),
             R"(
             Non-associative Mohr-Coulomb with linear isotropic strain hardening

             Implementation based on the book:
             de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen.
             Computational methods for plasticity: theory and applications.
             John Wiley & Sons, 2011.

             Example usage:
                >>> import pyroclastmpm as pm
                >>> mat = pm.MohrCoulomb(1000, 1e6, 0.25, 1e3, 30, 15, 1)
                >>> particles = pm.ParticlesContainer(np.array([0.,0.,0]))
                >>> particles = mat.stress_update(particles, 0)

             Parameters
             ----------
             density : float
                 Material density.
             E : float
                 Young's modulus.
             pois : float, optional
                 Poisson's ratio, by default 0.
             cohesion : float, optional
                 Cohesion, by default 0.
             friction_angle : float, optional
                 Friction angle (degrees), by default 0.
             dilatancy_angle : float, optional
                 Dilatancy angle (degrees), by default 0.
             H : float, optional
                 Linear hardening modulus, by default 1.

             )",
             py::arg("density"), py::arg("E"), py::arg("pois") = 0.0,
             py::arg("cohesion") = 0.0, py::arg("friction_angle") = 0.0,
             py::arg("dilatancy_angle") = 0.0, py::arg("H") = 0.0);
  material_cls.def(
      "stress_update",
      [](MohrCoulomb &self, ParticlesContainer particles_ref, int mat_id) {
        self.stress_update(particles_ref, mat_id);
        return std::make_tuple(particles_ref, mat_id);
      },
      R"(
        Perform a stress update step.

        Example usage:
            >>> import pyroclastmpm as pm
            >>> mat = pm.MohrCoulomb(1000, 1e6, 0.25, 1e3, 30, 15, 1)
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
      [](MohrCoulomb &self, ParticlesContainer particles_ref, int mat_id) {
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
      [](MohrCoulomb &self) {
        return std::vector<Matrixr>(self.eps_e_gpu.begin(),
                                    self.eps_e_gpu.end());
      },
      [](MohrCoulomb &self, const std::vector<Matrixr> &value) {
        cpu_array<Matrixr> host_val = value;
        self.eps_e_gpu = host_val;
      },
      "Elastic strain tensors (infinitesimal)");

  material_cls.def_readwrite("E", &MohrCoulomb::E, "Young's modulus");
  material_cls.def_readwrite("pois", &MohrCoulomb::pois, "Poisson's ratio");
  material_cls.def_readwrite("shear_modulus", &MohrCoulomb::shear_modulus,
                       "Shear modulus G");
  material_cls.def_readwrite("lame_modulus", &MohrCoulomb::lame_modulus,
                       "Lame modulus lambda");
  material_cls.def_readwrite("bulk_modulus", &MohrCoulomb::bulk_modulus,
                       "Bulk modulus K");
  material_cls.def_readwrite("density", &MohrCoulomb::density,
                       "Bulk density of the material");

  material_cls.def_readwrite("do_update_history", &MohrCoulomb::do_update_history,
                       "Flag if we update the history or not");
  material_cls.def_readwrite(
      "is_velgrad_strain_increment", &MohrCoulomb::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                       udpdate)");

  material_cls.def_readwrite("do_update_history", &MohrCoulomb::do_update_history,
                       "Flag if we update the history or not");
  material_cls.def_readwrite(
      "is_velgrad_strain_increment", &MohrCoulomb::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                       udpdate)");
                       
}

}