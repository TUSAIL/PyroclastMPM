#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/materials/linearelastic.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/materials/newtonfluid.cuh"
#include "pyroclastmpm/materials/localrheo.cuh"
#include "pyroclastmpm/materials/vonmises.cuh"

namespace py = pybind11;

namespace pyroclastmpm
{
  // py::class_<ParticlesContainer>(m, "ParticlesContainer")
  //     .def(py::init<std::vector<Vectorr>, std::vector<Vectorr>,
  //                   std::vector<int>, std::vector<bool>, std::vector<Matrix3r>, std::vector<Real>,
  //                   std::vector<Real>, std::vector<OutputType>>(),
  //          py::arg("positions"),
  //          py::arg("velocities") = std::vector<Vectorr>(),
  //          py::arg("colors") = std::vector<int>(),
  //          py::arg("is_rigid") = std::vector<bool>(),
  //          py::arg("stresses") = std::vector<Matrix3r>(),
  //          py::arg("masses") = std::vector<Real>(),
  //          py::arg("volumes") = std::vector<Real>(),
  //          py::arg("output_formats") = std::vector<OutputType>())
  //     .def("partition", &ParticlesContainer::partition)

  /**
   * @brief Create a pybind11 module for the materials module.
   *
   * @param m The pybind11 module to add the materials module to.
   */
  void materials_module(py::module &m)
  {
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
             [](LinearElastic &self, ParticlesContainer particles_ref, int mat_id)
             {
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
              // [ ] LinearElastic::calculate_timestep (TODO confirm correct formula and consistent with othe code)
              mat.bulk_modulus = t[6].cast<Real>();
              return mat;
            }));


    py::class_<VonMises>(m, "VonMises")
        .def(py::init<Real, Real, Real, Real, Real>(),
             py::arg("density"),
             py::arg("E"),
             py::arg("pois"),
             py::arg("yield_stress"),
             py::arg("H"))
        .def("stress_update",
             [](VonMises &self, ParticlesContainer particles_ref, int mat_id)
             {
               self.stress_update(particles_ref, mat_id);
               return std::make_tuple(particles_ref, mat_id);
             })
        .def("initialize",
             [](VonMises &self, ParticlesContainer particles_ref, int mat_id)
             {
               self.initialize(particles_ref, mat_id);
               return std::make_tuple(particles_ref, mat_id);
             })
        .def_readwrite("E", &VonMises::E)
        .def_readwrite("pois", &VonMises::pois)
        .def_readwrite("shear_modulus", &VonMises::shear_modulus)
        .def_readwrite("lame_modulus", &VonMises::lame_modulus)
        .def_readwrite("bulk_modulus", &VonMises::bulk_modulus)
        .def_readwrite("density", &VonMises::density)
        .def_readwrite("name", &VonMises::name);


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
             [](LocalGranularRheology &self, ParticlesContainer particles_ref, int mat_id)
             {
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
    // .def(
    //     "mp_benchmark",
    //     [](LocalGranularRheology &self,
    //     std::vector<Matrix3r> _stress_cpu,
    //     std::vector<uint8_t> _phases_cpu,
    //     std::vector<Matrixr> _velocity_gradient_cpu,
    //     std::vector<Real> _volume_cpu,
    //     std::vector<Real> _mass_cpu)
    //     {
    //       self.mp_benchmark( _stress_cpu, _phases_cpu, _velocity_gradient_cpu, _volume_cpu, _mass_cpu);
    //       return std::make_tuple(_stress_cpu, _phases_cpu);
    //     });
  };

} // namespace pyroclastmpm