#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

#include "pyroclastmpm/common/types_common.cuh"
// #include "pyroclastmpm/materials/materials.cuh"
#include "pyroclastmpm/particles/particles.cuh"

namespace py = pybind11;

namespace pyroclastmpm
{

  void particles_module(py::module &m)
  {
    py::class_<ParticlesContainer>(m, "ParticlesContainer")
        .def(py::init<std::vector<Vectorr>, std::vector<Vectorr>,
                      std::vector<int>, std::vector<bool>, std::vector<Matrix3r>, std::vector<Real>,
                      std::vector<Real>, std::vector<OutputType>>(),
             py::arg("positions"),
             py::arg("velocities") = std::vector<Vectorr>(),
             py::arg("colors") = std::vector<int>(),
             py::arg("is_rigid") = std::vector<bool>(),
             py::arg("stresses") = std::vector<Matrix3r>(),
             py::arg("masses") = std::vector<Real>(),
             py::arg("volumes") = std::vector<Real>(),
             py::arg("output_formats") = std::vector<OutputType>())
        .def("partition", &ParticlesContainer::partition)
        .def_readonly("num_particles",
                      &ParticlesContainer::num_particles) // NUM PARTICLES
        .def_property(
            "positions",
            [](ParticlesContainer &self)
            {
              return std::vector<Vectorr>(self.positions_gpu.begin(),
                                          self.positions_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Vectorr> &value)
            {
              cpu_array<Vectorr> host_val = value;
              self.positions_gpu = host_val;
            } // setter
            ) // POSITIONS
        .def_property(
            "velocities",
            [](ParticlesContainer &self)
            {
              return std::vector<Vectorr>(self.velocities_gpu.begin(),
                                          self.velocities_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Vectorr> &value)
            {
              cpu_array<Vectorr> host_val = value;
              self.velocities_gpu = value;
            } // setter
            ) // VELOCITIES
        .def_property(
            "stresses",
            [](ParticlesContainer &self)
            {
              return std::vector<Matrix3r>(self.stresses_gpu.begin(),
                                           self.stresses_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Matrix3r> &value)
            {
              cpu_array<Matrix3r> host_val = value;
              self.stresses_gpu = host_val;
            } // setter
            ) // STRESSES
        .def_property(
            "F",
            [](ParticlesContainer &self)
            {
              return std::vector<Matrixr>(self.F_gpu.begin(), self.F_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Matrixr> &value)
            {
              cpu_array<Matrixr> host_val = value;
              self.F_gpu = host_val;
            } // setter
            ) // DEFORMATION MATRIX
        .def_property(
            "velocity_gradient",
            [](ParticlesContainer &self)
            {
              return std::vector<Matrixr>(self.velocity_gradient_gpu.begin(),
                                          self.velocity_gradient_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Matrixr> &value)
            {
              cpu_array<Matrixr> host_val = value;
              self.velocity_gradient_gpu = value;
            } // setter
            ) // VELOCITY GRADIENT
        .def_property(
            "pressures",
            [](ParticlesContainer &self)
            {
              return std::vector<Real>(self.pressures_gpu.begin(),
                                       self.pressures_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Real> &value)
            {
              cpu_array<Real> host_val = value;
              self.pressures_gpu = host_val;
            } // setter
            ) // PRESSURES
        .def_property(
            "masses",
            [](ParticlesContainer &self)
            {
              return std::vector<Real>(self.masses_gpu.begin(),
                                       self.masses_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Real> &value)
            {
              cpu_array<Real> host_val = value;
              self.masses_gpu = host_val;
            } // setter
            ) // MASSES
        .def_property(
            "volumes",
            [](ParticlesContainer &self)
            {
              return std::vector<Real>(self.volumes_gpu.begin(),
                                       self.volumes_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Real> &value)
            {
              cpu_array<Real> host_val = value;
              self.volumes_gpu = host_val;
            } // setter
            ) // VOLUMES
        .def_property(
            "volumes_original",
            [](ParticlesContainer &self)
            {
              return std::vector<Real>(self.volumes_original_gpu.begin(),
                                       self.volumes_original_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<Real> &value)
            {
              cpu_array<Real> host_val = value;
              self.volumes_original_gpu = host_val;
            } // setter
            ) // VOLUMES ORIGINAL
        .def_property(
            "colors",
            [](ParticlesContainer &self)
            {
              return std::vector<int>(self.colors_gpu.begin(),
                                      self.colors_gpu.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<int> &value)
            {
              cpu_array<int> host_val = value;
              self.colors_gpu = host_val;
            } // setter
            ) // COLORS

        .def_property(
            "output_formats",
            [](ParticlesContainer &self)
            {
              return std::vector<OutputType>(self.output_formats.begin(),
                                      self.output_formats.end());
            }, // getter
            [](ParticlesContainer &self, const std::vector<OutputType> &value)
            {
              cpu_array<OutputType> host_val = value;
              self.output_formats = host_val;
            } // setter
            ) // COLORS
        .def(py::pickle(
            [](const ParticlesContainer &a) { // dump
              return py::make_tuple(
                  std::vector<Vectorr>(a.positions_gpu.begin(),
                                       a.positions_gpu.end()),
                  std::vector<Vectorr>(a.velocities_gpu.begin(),
                                       a.velocities_gpu.end()),
                  std::vector<int>(a.colors_gpu.begin(), a.colors_gpu.end()),
                  std::vector<int>(a.is_rigid_gpu.begin(), a.is_rigid_gpu.end()),
                  std::vector<Matrix3r>(a.stresses_gpu.begin(),
                                        a.stresses_gpu.end()),
                  std::vector<Real>(a.masses_gpu.begin(), a.masses_gpu.end()),
                  std::vector<Real>(a.volumes_gpu.begin(), a.volumes_gpu.end()),
                  std::vector<OutputType>(a.output_formats.begin(), a.output_formats.end()),
                  std::vector<Matrixr>(a.F_gpu.begin(), a.F_gpu.end()),
                  std::vector<Matrixr>(a.velocity_gradient_gpu.begin(),
                                       a.velocity_gradient_gpu.end()),
                  std::vector<Matrixr>(a.strain_increments_gpu.begin(),
                                       a.strain_increments_gpu.end()),
                  std::vector<Vectorr>(a.dpsi_gpu.begin(), a.dpsi_gpu.end()),
                  std::vector<Real>(a.volumes_original_gpu.begin(),
                                    a.volumes_original_gpu.end()),
                  std::vector<Real>(a.psi_gpu.begin(), a.psi_gpu.end()),
                  std::vector<Real>(a.densities_gpu.begin(),
                                    a.densities_gpu.end()),
                  std::vector<Real>(a.pressures_gpu.begin(),
                                    a.pressures_gpu.end()),
                  std::vector<int>(a.phases_gpu.begin(), a.phases_gpu.end()));

            },
            [](py::tuple t) { // load
              ParticlesContainer particles = ParticlesContainer(
                  t[0].cast<std::vector<Vectorr>>(),
                  t[1].cast<std::vector<Vectorr>>(),
                  t[2].cast<std::vector<int>>(),
                  t[3].cast<std::vector<bool>>(),
                  t[4].cast<std::vector<Matrix3r>>(),
                  t[5].cast<std::vector<Real>>(),
                  t[6].cast<std::vector<Real>>(),
                  t[7].cast<std::vector<OutputType>>());
              particles.F_gpu = t[8].cast<std::vector<Matrixr>>();
              particles.velocity_gradient_gpu =
                  t[9].cast<std::vector<Matrixr>>();
              particles.strain_increments_gpu =
                  t[10].cast<std::vector<Matrixr>>();
              particles.dpsi_gpu = t[11].cast<std::vector<Vectorr>>();
              particles.volumes_original_gpu = t[12].cast<std::vector<Real>>();
              particles.psi_gpu = t[13].cast<std::vector<Real>>();
              particles.densities_gpu = t[14].cast<std::vector<Real>>();
              particles.pressures_gpu = t[15].cast<std::vector<Real>>();
              particles.phases_gpu = t[16].cast<std::vector<int>>();
              return particles;
            }));
  };

} // namespace pyroclastmpm