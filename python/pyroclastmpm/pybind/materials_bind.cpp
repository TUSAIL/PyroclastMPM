#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/materials/linearelastic.h"
#include "pyroclastmpm/materials/localrheo.h"
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
void materials_module(py::module &m) {

  py::class_<Material>(m, "Material")
      .def(py::init<>())
      .def_readwrite("density", &Material::density)
      .def_readwrite("name", &Material::name)
      .def(py::pickle(
          [](const Material &a) { // dump
            return py::make_tuple(a.density, a.name);
          },
          [](py::tuple t) { // load
            return Material{t[0].cast<Real>(), t[1].cast<std::string>()};
          }));

  py::class_<LinearElastic>(m, "LinearElastic")
      .def(py::init<Real, Real, Real>(), py::arg("density"), py::arg("E"),
           py::arg("pois") = 0.)
      .def("stress_update",
           [](LinearElastic &self, ParticlesContainer particles_ref,
              int mat_id) {
             self.stress_update(particles_ref, mat_id);
             return std::make_tuple(particles_ref, mat_id);
           })
      .def_readwrite("E", &LinearElastic::E)
      .def_readwrite("pois", &LinearElastic::pois)
      .def_readwrite("shear_modulus", &LinearElastic::shear_modulus)
      .def_readwrite("lame_modulus", &LinearElastic::lame_modulus)
      .def_readwrite("bulk_modulus", &LinearElastic::bulk_modulus)
      .def_readwrite("density", &LinearElastic::density)
      .def_readwrite("name", &LinearElastic::name)
      .def(py::pickle(
          [](const LinearElastic &a) { // dump
            return py::make_tuple(a.density, a.E, a.pois, a.name,
                                  a.shear_modulus, a.lame_modulus,
                                  a.bulk_modulus);
          },
          [](py::tuple t) { // load
            LinearElastic mat = LinearElastic{
                t[0].cast<Real>(), t[1].cast<Real>(), t[2].cast<Real>()};
            mat.name = t[3].cast<std::string>();
            mat.shear_modulus = t[4].cast<Real>();
            mat.lame_modulus = t[5].cast<Real>();
            mat.bulk_modulus = t[6].cast<Real>();
            return mat;
          }));

  py::class_<VonMises>(m, "VonMises")
      .def(py::init<Real, Real, Real, Real, Real>(), py::arg("density"),
           py::arg("E"), py::arg("pois"), py::arg("yield_stress"), py::arg("H"))
      .def("stress_update",
           [](VonMises &self, ParticlesContainer particles_ref, int mat_id) {
             self.stress_update(particles_ref, mat_id);
             return std::make_tuple(particles_ref, mat_id);
           })
      .def("initialize",
           [](VonMises &self, ParticlesContainer particles_ref, int mat_id) {
             self.initialize(particles_ref, mat_id);
             return std::make_tuple(particles_ref, mat_id);
           }) // required for allocating memory to internal variables
      .def_property(
          "eps_e",
          [](VonMises &self) {
            return std::vector<Matrixr>(self.eps_e_gpu.begin(),
                                        self.eps_e_gpu.end());
          }, // getter
          [](VonMises &self, const std::vector<Matrixr> &value) {
            cpu_array<Matrixr> host_val = value;
            self.eps_e_gpu = host_val;
          } // setter
          ) // elastic strain (infinitesimal)
      .def_readwrite("E", &VonMises::E)
      .def_readwrite("pois", &VonMises::pois)
      .def_readwrite("shear_modulus", &VonMises::shear_modulus)
      .def_readwrite("lame_modulus", &VonMises::lame_modulus)
      .def_readwrite("bulk_modulus", &VonMises::bulk_modulus)
      .def_readwrite("density", &VonMises::density)
      .def_readwrite("name", &VonMises::name);

  py::class_<MohrCoulomb>(m, "MohrCoulomb")
      .def(py::init<Real, Real, Real, Real, Real, Real, Real>(),
           py::arg("density"), py::arg("E"), py::arg("pois"),
           py::arg("cohesion"), py::arg("friction_angle"),
           py::arg("dilatancy_angle"), py::arg("H"))
      .def("stress_update",
           [](MohrCoulomb &self, ParticlesContainer particles_ref, int mat_id) {
             self.stress_update(particles_ref, mat_id);
             return std::make_tuple(particles_ref, mat_id);
           })
      .def("initialize",
           [](MohrCoulomb &self, ParticlesContainer particles_ref, int mat_id) {
             self.initialize(particles_ref, mat_id);
             return std::make_tuple(particles_ref, mat_id);
           }) // required for allocating memory to internal variables
      .def_property(
          "eps_e",
          [](MohrCoulomb &self) {
            return std::vector<Matrixr>(self.eps_e_gpu.begin(),
                                        self.eps_e_gpu.end());
          }, // getter
          [](MohrCoulomb &self, const std::vector<Matrixr> &value) {
            cpu_array<Matrixr> host_val = value;
            self.eps_e_gpu = host_val;
          } // setter
          ) // elastic strain (infinitesimal)
      .def_readwrite("E", &MohrCoulomb::E)
      .def_readwrite("pois", &MohrCoulomb::pois)
      .def_readwrite("shear_modulus", &MohrCoulomb::shear_modulus)
      .def_readwrite("lame_modulus", &MohrCoulomb::lame_modulus)
      .def_readwrite("bulk_modulus", &MohrCoulomb::bulk_modulus)
      .def_readwrite("density", &MohrCoulomb::density)
      .def_readwrite("name", &MohrCoulomb::name);

  py::class_<NewtonFluid>(m, "NewtonFluid")
      .def(py::init<Real, Real, Real, Real>(), py::arg("density"),
           py::arg("viscosity"), py::arg("bulk_modulus") = 0.,
           py::arg("gamma") = 7.)
      .def_readwrite("density", &NewtonFluid::density)
      .def_readwrite("name", &NewtonFluid::name)
      .def_readwrite("viscosity", &NewtonFluid::viscosity)
      .def_readwrite("bulk_modulus", &NewtonFluid::bulk_modulus)
      .def_readwrite("gamma", &NewtonFluid::gamma)
      .def(py::pickle(
          [](const NewtonFluid &a) { // dump
            return py::make_tuple(a.density, a.viscosity, a.bulk_modulus,
                                  a.gamma, a.name);
          },
          [](py::tuple t) { // load
            NewtonFluid mat = NewtonFluid{t[0].cast<Real>(), t[1].cast<Real>(),
                                          t[2].cast<Real>(), t[3].cast<Real>()};
            mat.name = t[4].cast<std::string>();
            return mat;
          }));

  py::class_<LocalGranularRheology>(m, "LocalGranularRheology")
      .def(py::init<Real, Real, Real, Real, Real, Real, Real, Real, Real>(),
           py::arg("density"), py::arg("E"), py::arg("pois"), py::arg("I0"),
           py::arg("mu_s"), py::arg("mu_2"), py::arg("rho_c"),
           py::arg("particle_diameter"), py::arg("particle_density"))
      .def("stress_update",
           [](LocalGranularRheology &self, ParticlesContainer particles_ref,
              int mat_id) {
             self.stress_update(particles_ref, mat_id);
             return std::make_tuple(particles_ref, mat_id);
           })
      .def_readwrite("density", &LocalGranularRheology::density)
      .def_readwrite("name", &LocalGranularRheology::name)
      .def_readwrite("E", &LocalGranularRheology::E)
      .def_readwrite("pois", &LocalGranularRheology::pois)
      .def_readwrite("shear_modulus", &LocalGranularRheology::shear_modulus)
      .def_readwrite("lame_modulus", &LocalGranularRheology::lame_modulus)
      .def_readwrite("bulk_modulus", &LocalGranularRheology::bulk_modulus)
      .def_readwrite("mu_s", &LocalGranularRheology::mu_s)
      .def_readwrite("mu_2", &LocalGranularRheology::mu_2)
      .def_readwrite("I0", &LocalGranularRheology::I0)
      .def_readwrite("rho_c", &LocalGranularRheology::rho_c)
      .def_readwrite("EPS", &LocalGranularRheology::EPS)
      .def_readwrite("particle_density",
                     &LocalGranularRheology::particle_density)
      .def_readwrite("particle_diameter",
                     &LocalGranularRheology::particle_diameter)
      .def(py::pickle(
          [](const LocalGranularRheology &a) { // dump
            return py::make_tuple(a.density, a.E, a.pois, a.I0, a.mu_s, a.mu_2,
                                  a.rho_c, a.particle_diameter,
                                  a.particle_density, a.name, a.shear_modulus,
                                  a.lame_modulus, a.bulk_modulus, a.EPS);
          },
          [](py::tuple t) { // load
            LocalGranularRheology mat = LocalGranularRheology{
                t[0].cast<Real>(), t[1].cast<Real>(), t[2].cast<Real>(),
                t[3].cast<Real>(), t[4].cast<Real>(), t[5].cast<Real>(),
                t[6].cast<Real>(), t[7].cast<Real>(), t[8].cast<Real>()};
            mat.name = t[9].cast<std::string>();
            mat.shear_modulus = t[10].cast<Real>();
            mat.lame_modulus = t[11].cast<Real>();
            mat.bulk_modulus = t[12].cast<Real>();
            mat.EPS = t[13].cast<Real>();
            return mat;
          }));
};

} // namespace pyroclastmpm