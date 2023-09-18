// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/materials/linearelastic.h"
#include "pyroclastmpm/materials/localrheo.h"
#include "pyroclastmpm/materials/modifiedcamclay.h"
#include "pyroclastmpm/materials/mohrcoulomb.h"
#include "pyroclastmpm/materials/newtonfluid.h"
#include "pyroclastmpm/materials/vonmises.h"
#include "pyroclastmpm/nodes/nodes.h"

namespace py = pybind11;

namespace pyroclastmpm {

/**
 * @brief Bindings for materials
 *
 * @param m pybind11 module
 */
void materials_module(const py::module &m) {

  /* Material Base */
  py::class_<Material> M_cls(m, "Material", py::dynamic_attr());
  M_cls.def(py::init<>(), "Material base class");
  M_cls.def_readwrite("density", &Material::density,
                      "Bulk density of the material");
  M_cls.def(
      py::pickle([](const Material &a) { return py::make_tuple(a.density); },
                 [](py::tuple t) { return Material{t[0].cast<Real>()}; }),
      "Material pickle");

  /* Linear Elastic */
  py::class_<LinearElastic> LE_cls(m, "LinearElastic", py::dynamic_attr());
  LE_cls.def(py::init<Real, Real, Real>(),
             R"(
             Isotropic linear elastic material (infinitesimal strain)
             
             The stress tensor is given by:

             .. math::
                \boldsymbol{\sigma}= K\boldsymbol{I}\varepsilon_v + 2G\boldsymbol{\varepsilon}_s


             where :math:`G`, :math:`K` are the shear and bulk moduli respectively. The volumetric
             strain is :math:`\boldsymbol{\varepsilon}_v = \text{tr}(\boldsymbol{\varepsilon})` and the deviatoric
             strain is :math:`\boldsymbol{\varepsilon}_s = \boldsymbol{\varepsilon} - \frac{1}{3}\boldsymbol{I}\boldsymbol{\varepsilon}_v)`.
              


             Example usage:
                    >>> import pyroclastmpm as pm
                    >>> mat = pm.LinearElastic(1000, 1e6, 0.3)

             Parameters
             ----------
             density : float
                  Material density
             E : float
                  Young's modulus
             pois : float, optional
                  Poisson's ratio, by default 0
            )",
             py::arg("density"), py::arg("E"), py::arg("pois") = 0.);
  LE_cls.def(
      "stress_update",
      [](LinearElastic &self, ParticlesContainer particles_ref, int mat_id) {
        self.stress_update(particles_ref, mat_id);
        return std::make_tuple(particles_ref, mat_id);
      },
      R"(
        Perform a stress update step.

        Example usage:
            >>> import pyroclastmpm as pm
            >>> mat = pm.LinearElastic(1000, 1e6, 0.3)
            >>> particles = pm.ParticlesContainer(np.array([0.,0.,0]))
            >>> particles = mat.stress_update(particles, 0)
        
        Returns
        -------
        ParticlesContainer
            Particle container (updated stress)
      )");

  LE_cls.def_readwrite("E", &LinearElastic::E, "Young's modulus");
  LE_cls.def_readwrite("pois", &LinearElastic::pois, "Poisson's ratio");
  LE_cls.def_readwrite("shear_modulus", &LinearElastic::shear_modulus,
                       "Shear modulus G");
  LE_cls.def_readwrite("lame_modulus", &LinearElastic::lame_modulus,
                       "Lame modulus lambda");
  LE_cls.def_readwrite("bulk_modulus", &LinearElastic::bulk_modulus,
                       "Bulk modulus (K)");
  LE_cls.def_readwrite("density", &LinearElastic::density,
                       "Bulk density of the material");

  LE_cls.def(py::pickle(
                 [](const LinearElastic &a) {
                   return py::make_tuple(a.density, a.E, a.pois,
                                         a.shear_modulus, a.lame_modulus,
                                         a.bulk_modulus);
                 },
                 [](py::tuple t) {
                   auto mat = LinearElastic{
                       t[0].cast<Real>(), t[1].cast<Real>(), t[2].cast<Real>()};
                   mat.shear_modulus = t[3].cast<Real>();
                   mat.lame_modulus = t[4].cast<Real>();
                   mat.bulk_modulus = t[5].cast<Real>();
                   return mat;
                 }),
             "Pickling for LinearElastic");

  /* Modified Cam Clay */
  py::class_<ModifiedCamClay> MCC_cls(m, "ModifiedCamClay", py::dynamic_attr());
  MCC_cls.def(
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

  MCC_cls.def(
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
  MCC_cls.def(
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
  MCC_cls.def_property(
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
  MCC_cls.def_readwrite("E", &ModifiedCamClay::E, "Young's modulus");
  MCC_cls.def_readwrite("pois", &ModifiedCamClay::pois, "Poisson's ratio");
  MCC_cls.def_readwrite("shear_modulus", &ModifiedCamClay::shear_modulus,
                        "Shear modulus G");
  MCC_cls.def_readwrite("lame_modulus", &ModifiedCamClay::lame_modulus,
                        "Lame modulus lambda");
  MCC_cls.def_readwrite("bulk_modulus", &ModifiedCamClay::bulk_modulus,
                        "Bulk modulus K");
  MCC_cls.def_readwrite("density", &ModifiedCamClay::density,
                        "Bulk density of the material");
  MCC_cls.def_readwrite("do_update_history",
                        &ModifiedCamClay::do_update_history,
                        "Flag if we update the history or not");
  MCC_cls.def_readwrite(
      "is_velgrad_strain_increment",
      &ModifiedCamClay::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                               udpdate)");
  MCC_cls.def("calculate_timestep", &ModifiedCamClay::calculate_timestep,
              "calculate_timestep");

  /* Von Mises */
  py::class_<VonMises> VM_cls(m, "VonMises", py::dynamic_attr());
  VM_cls.def(py::init<Real, Real, Real, Real, Real>(),
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
  VM_cls.def(
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
  VM_cls.def(
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
  VM_cls.def_property(
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
  VM_cls.def_readwrite("E", &VonMises::E, "Young's modulus");
  VM_cls.def_readwrite("pois", &VonMises::pois, "Poisson's ratio");
  VM_cls.def_readwrite("shear_modulus", &VonMises::shear_modulus,
                       "Shear modulus G");
  VM_cls.def_readwrite("lame_modulus", &VonMises::lame_modulus,
                       "Lame modulus lambda");
  VM_cls.def_readwrite("bulk_modulus", &VonMises::bulk_modulus,
                       "Bulk modulus K");
  VM_cls.def_readwrite("density", &VonMises::density,
                       "Bulk density of the material");
  VM_cls.def_readwrite("do_update_history", &VonMises::do_update_history,
                       "Flag if we update the history or not");
  VM_cls.def_readwrite(
      "is_velgrad_strain_increment", &VonMises::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                       udpdate)");
  /*Mohr Coulomb*/
  py::class_<MohrCoulomb> MC_cls(m, "MohrCoulomb");
  MC_cls.def(py::init<Real, Real, Real, Real, Real, Real, Real>(),
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
  MC_cls.def(
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
  MC_cls.def(
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
  MC_cls.def_property(
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

  MC_cls.def_readwrite("E", &MohrCoulomb::E, "Young's modulus");
  MC_cls.def_readwrite("pois", &MohrCoulomb::pois, "Poisson's ratio");
  MC_cls.def_readwrite("shear_modulus", &MohrCoulomb::shear_modulus,
                       "Shear modulus G");
  MC_cls.def_readwrite("lame_modulus", &MohrCoulomb::lame_modulus,
                       "Lame modulus lambda");
  MC_cls.def_readwrite("bulk_modulus", &MohrCoulomb::bulk_modulus,
                       "Bulk modulus K");
  MC_cls.def_readwrite("density", &MohrCoulomb::density,
                       "Bulk density of the material");

  MC_cls.def_readwrite("do_update_history", &MohrCoulomb::do_update_history,
                       "Flag if we update the history or not");
  MC_cls.def_readwrite(
      "is_velgrad_strain_increment", &MohrCoulomb::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                       udpdate)");

  MC_cls.def_readwrite("do_update_history", &MohrCoulomb::do_update_history,
                       "Flag if we update the history or not");
  MC_cls.def_readwrite(
      "is_velgrad_strain_increment", &MohrCoulomb::is_velgrad_strain_increment,
      R"(Flag if we should use strain increment instead of velocity gradient for constitutive
                       udpdate)");

  /* Newton Fluid */
  py::class_<NewtonFluid> NF_cls(m, "NewtonFluid", py::dynamic_attr());
  NF_cls.def(py::init<Real, Real, Real, Real>(),
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
  NF_cls.def_readwrite("density", &NewtonFluid::density,
                       "Bulk density of the material");
  NF_cls.def_readwrite("viscosity", &NewtonFluid::viscosity, "Viscosity");
  NF_cls.def_readwrite("bulk_modulus", &NewtonFluid::bulk_modulus,
                       "Bulk modulus K");
  NF_cls.def_readwrite("gamma", &NewtonFluid::gamma, "7 water and 1.4 for air");
  NF_cls.def(py::pickle(
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

  /* Local Rheology */
  py::class_<LocalGranularRheology> LGR_cls(m, "LocalGranularRheology",
                                            py::dynamic_attr());
  LGR_cls.def(py::init<Real, Real, Real, Real, Real, Real, Real, Real, Real>(),
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
  LGR_cls.def_readwrite("density", &LocalGranularRheology::density,
                        "Bulk density of the material");
  LGR_cls.def_readwrite("E", &LocalGranularRheology::E, "Young's modulus");
  LGR_cls.def_readwrite("pois", &LocalGranularRheology::pois,
                        "Poisson's ratio");
  LGR_cls.def_readwrite("shear_modulus", &LocalGranularRheology::shear_modulus,
                        "Shear modulus G");
  LGR_cls.def_readwrite("lame_modulus", &LocalGranularRheology::lame_modulus,
                        "Lame modulus lambda");
  LGR_cls.def_readwrite("bulk_modulus", &LocalGranularRheology::bulk_modulus,
                        "Bulk modulus K");
  LGR_cls.def_readwrite("mu_s", &LocalGranularRheology::mu_s,
                        "Critical friction angle (max)");
  LGR_cls.def_readwrite("mu_2", &LocalGranularRheology::mu_2,
                        "Critical friction angle (min)");
  LGR_cls.def_readwrite("I0", &LocalGranularRheology::I0, "Inertial number");
  LGR_cls.def_readwrite("rho_c", &LocalGranularRheology::rho_c,
                        "Critical density");
  LGR_cls.def_readwrite("EPS", &LocalGranularRheology::EPS, "EPS");
  LGR_cls.def_readwrite("particle_density",
                        &LocalGranularRheology::particle_density,
                        "Particle density");
  LGR_cls.def_readwrite("particle_diameter",
                        &LocalGranularRheology::particle_diameter,
                        "Particle diameter");
  LGR_cls.def(
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
};

} // namespace pyroclastmpm