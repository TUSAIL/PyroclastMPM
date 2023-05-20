#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/materials/linearelastic.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
// #include "pyroclastmpm/materials/newtonfluid/newtonfluidmat.cuh"
// #include "pyroclastmpm/materials/localrheo/localrheomat.cuh"
// #include "pyroclastmpm/materials/druckerprager/druckerpragermat.cuh"

namespace py = pybind11;

namespace pyroclastmpm
{

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

    // py::class_<NewtonFluid>(m, "NewtonFluid")
    //     .def(py::init<Real, Real, Real, Real>(), py::arg("density"),
    //          py::arg("viscosity"), py::arg("bulk_modulus") = 0.,
    //          py::arg("gamma") = 7.)
    //     .def_readwrite("density", &NewtonFluid::density)
    //     .def_readwrite("name", &NewtonFluid::name)
    //     .def_readwrite("viscosity", &NewtonFluid::viscosity)
    //     .def_readwrite("bulk_modulus", &NewtonFluid::bulk_modulus)
    //     .def_readwrite("gamma", &NewtonFluid::gamma)
    //     .def(py::pickle(
    //         [](const NewtonFluid &a) { // dump
    //           return py::make_tuple(a.density, a.viscosity, a.bulk_modulus,
    //                                 a.gamma, a.name);
    //         },
    //         [](py::tuple t) { // load
    //           NewtonFluid mat = NewtonFluid{t[0].cast<Real>(), t[1].cast<Real>(),
    //                                         t[2].cast<Real>(), t[3].cast<Real>()};
    //           mat.name = t[4].cast<std::string>();
    //           return mat;
    //         }));

    // py::class_<LocalGranularRheology>(m, "LocalGranularRheology")
    //     .def(py::init<Real, Real, Real, Real, Real, Real, Real, Real, Real>(),
    //          py::arg("density"), py::arg("E"), py::arg("pois"), py::arg("I0"),
    //          py::arg("mu_s"), py::arg("mu_2"), py::arg("rho_c"),
    //          py::arg("particle_diameter"), py::arg("particle_density"))
    //     .def_readwrite("density", &LocalGranularRheology::density)
    //     .def_readwrite("name", &LocalGranularRheology::name)
    //     .def_readwrite("E", &LocalGranularRheology::E)
    //     .def_readwrite("pois", &LocalGranularRheology::pois)
    //     .def_readwrite("shear_modulus", &LocalGranularRheology::shear_modulus)
    //     .def_readwrite("lame_modulus", &LocalGranularRheology::lame_modulus)
    //     .def_readwrite("bulk_modulus", &LocalGranularRheology::bulk_modulus)
    //     .def_readwrite("mu_s", &LocalGranularRheology::mu_s)
    //     .def_readwrite("mu_2", &LocalGranularRheology::mu_2)
    //     .def_readwrite("I0", &LocalGranularRheology::I0)
    //     .def_readwrite("rho_c", &LocalGranularRheology::rho_c)
    //     .def_readwrite("EPS", &LocalGranularRheology::EPS)
    //     .def_readwrite("particle_density",
    //                    &LocalGranularRheology::particle_density)
    //     .def_readwrite("particle_diameter",
    //                    &LocalGranularRheology::particle_diameter)
    //     .def(py::pickle(
    //         [](const LocalGranularRheology &a) { // dump
    //           return py::make_tuple(a.density, a.E, a.pois, a.I0, a.mu_s, a.mu_2,
    //                                 a.rho_c, a.particle_diameter,
    //                                 a.particle_density, a.name, a.shear_modulus,
    //                                 a.lame_modulus, a.bulk_modulus, a.EPS);
    //         },
    //         [](py::tuple t) { // load
    //           LocalGranularRheology mat = LocalGranularRheology{
    //               t[0].cast<Real>(), t[1].cast<Real>(), t[2].cast<Real>(),
    //               t[3].cast<Real>(), t[4].cast<Real>(), t[5].cast<Real>(),
    //               t[6].cast<Real>(), t[7].cast<Real>(), t[8].cast<Real>()};
    //           mat.name = t[9].cast<std::string>();
    //           mat.shear_modulus = t[10].cast<Real>();
    //           mat.lame_modulus = t[11].cast<Real>();
    //           mat.bulk_modulus = t[12].cast<Real>();
    //           mat.EPS = t[13].cast<Real>();
    //           return mat;
    //         }))
    //     .def(
    //         "mp_benchmark",
    //         [](LocalGranularRheology &self,
    //         std::vector<Matrix3r> _stress_cpu,
    //         std::vector<uint8_t> _phases_cpu,
    //         std::vector<Matrixr> _velocity_gradient_cpu,
    //         std::vector<Real> _volume_cpu,
    //         std::vector<Real> _mass_cpu)
    //         {
    //           self.mp_benchmark( _stress_cpu, _phases_cpu, _velocity_gradient_cpu, _volume_cpu, _mass_cpu);
    //           return std::make_tuple(_stress_cpu, _phases_cpu);
    //         });

    //         py::class_<DruckerPrager>(m, "DruckerPrager")
    //     .def(py::init<Real, Real, Real, Real, Real, Real>(),
    //          py::arg("density"), py::arg("E"), py::arg("pois"), py::arg("friction_angle"),
    //          py::arg("cohesion"), py::arg("vcs"))
    //     .def_readwrite("density", &DruckerPrager::density)
    //     .def_readwrite("name", &DruckerPrager::name)
    //     .def_readwrite("E", &DruckerPrager::E)
    //     .def_readwrite("pois", &DruckerPrager::pois)
    //     .def_readwrite("shear_modulus", &DruckerPrager::shear_modulus)
    //     .def_readwrite("lame_modulus", &DruckerPrager::lame_modulus)
    //     .def_readwrite("friction_angle", &DruckerPrager::friction_angle)
    //     .def_readwrite("cohesion", &DruckerPrager::cohesion)
    //     .def_readwrite("vcs", &DruckerPrager::vcs)
    //     .def(
    //         "py_stress_update",
    //         [](DruckerPrager &self, Matrix3r stress, Matrix3r Fe, Real logJp, Matrix3r Fp_tr, Real alpha, const int dim)
    //         {
    //           self.outbound_stress_update(stress, Fe, logJp, Fp_tr, alpha, dim);
    //           return std::make_tuple(stress, Fe, logJp);
    //         })

    //     .def(py::pickle(
    //         [](const DruckerPrager &a) { // dump
    //           return py::make_tuple(a.density, a.E, a.pois, a.friction_angle, a.cohesion, a.vcs,
    //                                 a.name);
    //         },
    //         [](py::tuple t) { // load
    //           DruckerPrager mat = DruckerPrager{t[0].cast<Real>(), t[1].cast<Real>(),
    //                                             t[2].cast<Real>(), t[3].cast<Real>(),
    //                                             t[4].cast<Real>(), t[5].cast<Real>()};
    //           mat.name = t[6].cast<std::string>();
    //           return mat;
    //         }));

    // py::class_<NonLocalNGF>(m, "NonLocalNGF")
    //     .def(py::init<Real, Real, Real, Real, Real, Real, Real, Real, Real,
    //                   Real>(),
    //          py::arg("density"), py::arg("E"), py::arg("pois"), py::arg("I0"),
    //          py::arg("mu_s"), py::arg("mu_2"), py::arg("rho_c"),
    //          py::arg("particle_diameter"), py::arg("particle_density"),
    //          py::arg("A"))
    //     .def_readwrite("density", &NonLocalNGF::density)
    //     .def_readwrite("name", &NonLocalNGF::name)
    //     .def_readwrite("E", &NonLocalNGF::E)
    //     .def_readwrite("pois", &NonLocalNGF::pois)
    //     .def_readwrite("shear_modulus", &NonLocalNGF::shear_modulus)
    //     .def_readwrite("lame_modulus", &NonLocalNGF::lame_modulus)
    //     .def_readwrite("bulk_modulus", &NonLocalNGF::bulk_modulus)
    //     .def_readwrite("mu_s", &NonLocalNGF::mu_s)
    //     .def_readwrite("mu_2", &NonLocalNGF::mu_2)
    //     .def_readwrite("I0", &NonLocalNGF::I0)
    //     .def_readwrite("rho_c", &NonLocalNGF::rho_c)
    //     .def_readwrite("EPS", &NonLocalNGF::EPS)
    //     .def_readwrite("A", &NonLocalNGF::A)
    //     .def_readwrite("particle_density", &NonLocalNGF::particle_density)
    //     .def_readwrite("particle_diameter",
    //                    &NonLocalNGF::particle_diameter)
    //     .def(py::pickle(
    //         [](const NonLocalNGF& a) {  // dump
    //           return py::make_tuple(
    //               a.density, a.E, a.pois, a.I0, a.mu_s, a.mu_2, a.rho_c,
    //               a.particle_diameter, a.particle_density, a.A, a.name,
    //               a.shear_modulus, a.lame_modulus, a.bulk_modulus);

    //         },
    //         [](py::tuple t) {  // load
    //           NonLocalNGF mat = NonLocalNGF{t[0].cast<Real>(), t[1].cast<Real>(),
    //                                         t[2].cast<Real>(), t[3].cast<Real>(),
    //                                         t[4].cast<Real>(), t[5].cast<Real>(),
    //                                         t[6].cast<Real>(), t[7].cast<Real>(),
    //                                         t[8].cast<Real>(), t[9].cast<Real>()};
    //           mat.name = t[10].cast<std::string>();
    //           mat.shear_modulus = t[11].cast<Real>();
    //           mat.lame_modulus = t[12].cast<Real>();
    //           mat.bulk_modulus = t[13].cast<Real>();
    //           return mat;
    //         }));
  };

} // namespace pyroclastmpm