#include "pyroclastmpm/materials/linearelastic.h"
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyroclastmpm/common/types_common.h"


namespace py = pybind11;


namespace pyroclastmpm {

void linearelastic_module(const py::module &m) {
 /* Linear Elastic */
  py::class_<LinearElastic> material_cls(m, "LinearElastic", py::dynamic_attr());
  material_cls.def(py::init<Real, Real, Real>(),
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
  material_cls.def(
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

  material_cls.def_readwrite("E", &LinearElastic::E, "Young's modulus");
  material_cls.def_readwrite("pois", &LinearElastic::pois, "Poisson's ratio");
  material_cls.def_readwrite("shear_modulus", &LinearElastic::shear_modulus,
                       "Shear modulus G");
  material_cls.def_readwrite("lame_modulus", &LinearElastic::lame_modulus,
                       "Lame modulus lambda");
  material_cls.def_readwrite("bulk_modulus", &LinearElastic::bulk_modulus,
                       "Bulk modulus (K)");
  material_cls.def_readwrite("density", &LinearElastic::density,
                       "Bulk density of the material");

  material_cls.def(py::pickle(
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



}

}