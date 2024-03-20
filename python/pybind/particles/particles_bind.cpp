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
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/particles/particles.h"

namespace py = pybind11;

namespace pyroclastmpm
{

  py::tuple pickle_save_particles(const ParticlesContainer &particles)
  {
    return py::make_tuple(
        std::vector<Vectorr>(particles.positions_gpu.begin(),
                             particles.positions_gpu.end()), // 0
        std::vector<Vectorr>(particles.velocities_gpu.begin(),
                             particles.velocities_gpu.end()), // 1
        std::vector<int>(particles.colors_gpu.begin(),
                         particles.colors_gpu.end()), // 2
        std::vector<int>(particles.is_rigid_gpu.begin(),
                         particles.is_rigid_gpu.end()), // 3
        std::vector<Matrix3r>(particles.stresses_gpu.begin(),
                              particles.stresses_gpu.end()), // 4
        std::vector<Real>(particles.masses_gpu.begin(),
                          particles.masses_gpu.end()), // 5
        std::vector<Real>(particles.volumes_gpu.begin(),
                          particles.volumes_gpu.end()), // 6
        std::vector<std::string>(particles.output_formats.begin(),
                                 particles.output_formats.end()), // 7
        std::vector<Matrixr>(particles.F_gpu.begin(), particles.F_gpu.end()),
        std::vector<Matrixr>(particles.velocity_gradient_gpu.begin(),
                             particles.velocity_gradient_gpu.end()),
        std::vector<Vectorr>(particles.dpsi_gpu.begin(),
                             particles.dpsi_gpu.end()),
        std::vector<Real>(particles.volumes_original_gpu.begin(),
                          particles.volumes_original_gpu.end()),
        std::vector<Real>(particles.psi_gpu.begin(), particles.psi_gpu.end()));
  }

  ParticlesContainer pickle_load_particles(py::tuple t)
  {
    auto particles = ParticlesContainer(
        t[0].cast<std::vector<Vectorr>>(), t[1].cast<std::vector<Vectorr>>(),
        t[2].cast<std::vector<int>>(), t[3].cast<std::vector<bool>>());
    particles.stresses_gpu = t[4].cast<std::vector<Matrix3r>>();
    particles.masses_gpu = t[5].cast<std::vector<Real>>();
    particles.volumes_gpu = t[6].cast<std::vector<Real>>();
    particles.output_formats = t[7].cast<std::vector<std::string>>();
    particles.F_gpu = t[8].cast<std::vector<Matrixr>>();
    particles.velocity_gradient_gpu = t[9].cast<std::vector<Matrixr>>();
    particles.dpsi_gpu = t[10].cast<std::vector<Vectorr>>();
    particles.volumes_original_gpu = t[11].cast<std::vector<Real>>();
    particles.psi_gpu = t[12].cast<std::vector<Real>>();
    return particles;
  }

  void particles_module(const py::module &m)
  {
    py::class_<ParticlesContainer> P_cls(
        m,
        // ("ParticlesContainer" + std::to_string(DIM) + "D").c_str(),
        "ParticlesContainer",
        py::dynamic_attr());

    P_cls.def(py::init<std::vector<Vectorr>,
                       std::vector<Vectorr>,
                       std::vector<int>,
                       std::vector<Vectorr>,
                       std::vector<Vectorr>,
                       std::vector<int>>(),
              R"(
    Stores the particles (Lagrangian markers) of the MPM simulation.

    Rigid particles are added to the simulation as normal particles. However,
    a mask has to be set to indicate which particle is rigid.


    .. highlight:: python
    .. code-block:: python
    
        import pyroclastmpm as pm
        import numpy as np
        positions = np.array([[0,0,0],[1,1,1]])
        velocities = np.array([[0,0,0],[1,1,1]])
        particles = pm.ParticlesContainer(positions,velocities)

    Parameters
    -----------
    positions : np.ndarray
        Positions of the particles
    velocities : np.ndarray, optional
        Velocities of the particles, by default None
    colors : List[int], optional
        A list (or np.array) of material type of the particles, and by default None
    rigid_positions : List[bool], optional
        Positions of the rigid particles, by default None
    rigid_velocities : List[bool], optional
        Velocities of the rigid particles, by default None
    rigid_colors : List[int], optional
        Material types of the rigid particles, and by default None

  )",
              py::arg("positions"),
              py::arg("velocities") = std::vector<Vectorr>(),
              py::arg("colors") = std::vector<int>() = std::vector<int>(),
              py::arg("rigid_positions") = std::vector<Vectorr>(),
              py::arg("rigid_velocities") = std::vector<Vectorr>(),
              py::arg("rigid_colors") = std::vector<int>());

    P_cls.def("set_output_formats", &ParticlesContainer::set_output_formats,
              R"(
            Set a list of formats that are outputted in aVect
            directory specified in :func:`set_globals`.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> particles = pm.ParticlesContainer(...)
                  >>> particles.set_output_formats(["vtk","csv"])
            
            Parameters
            ----------
            output_formats : List[str]
                List of output formats.

            )",
              py::arg("output_formats"));

    P_cls.def("output_vtk", &ParticlesContainer::output_vtk,
              R"(
            Output the data loaded in the particles container to a VTK file
            in a directory specified in :func:`set_globals`.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> particles = pm.ParticlesContainer(...)
                  >>> particles.output_vtk()

            )");

    P_cls.def("set_spawner", &ParticlesContainer::set_spawner,
              "Warning unstable");

    P_cls.def_readwrite(
        "exclude_rigid_from_output", &ParticlesContainer::exclude_rigid_from_output,
        "Option to eclude rigid particles from output");

    P_cls.def_readonly(
        "num_particles", &ParticlesContainer::num_particles,
        "Number of particles in the container (rigid and non-rigid)");

    P_cls.def_property(
        "positions",
        [](ParticlesContainer &self)
        {
          return std::vector<Vectorr>(self.positions_gpu.begin(),
                                      self.positions_gpu.end());
        },
        [](ParticlesContainer &self, const std::vector<Vectorr> &value)
        {
          cpu_array<Vectorr> host_val = value;
          self.positions_gpu = host_val;
        },
        "Positions of the particles");

    P_cls.def_property(
        "velocities",
        [](ParticlesContainer &self)
        {
          return std::vector<Vectorr>(self.velocities_gpu.begin(),
                                      self.velocities_gpu.end());
        },
        [](ParticlesContainer &self, const std::vector<Vectorr> &value)
        {
          cpu_array<Vectorr> host_val = value;
          self.velocities_gpu = value;
        },
        "Velocities of the particles");

    P_cls.def_property(
        "stresses",
        [](ParticlesContainer &self)
        {
          return std::vector<Matrix3r>(self.stresses_gpu.begin(),
                                       self.stresses_gpu.end());
        },
        [](ParticlesContainer &self, const std::vector<Matrix3r> &value)
        {
          cpu_array<Matrix3r> host_val = value;
          self.stresses_gpu = host_val;
        },
        "Stress fields");

    P_cls.def_property(
        "F",
        [](ParticlesContainer &self)
        {
          return std::vector<Matrixr>(self.F_gpu.begin(), self.F_gpu.end());
        },
        [](ParticlesContainer &self, const std::vector<Matrixr> &value)
        {
          cpu_array<Matrixr> host_val = value;
          self.F_gpu = host_val;
        },
        "Deformation gradients");

    P_cls.def_property(
        "velocity_gradient",
        [](ParticlesContainer &self)
        {
          return std::vector<Matrixr>(self.velocity_gradient_gpu.begin(),
                                      self.velocity_gradient_gpu.end());
        },
        [](ParticlesContainer &self, const std::vector<Matrixr> &value)
        {
          cpu_array<Matrixr> host_val = value;
          self.velocity_gradient_gpu = value;
        },
        "Velocity gradients");

    P_cls.def_property(
        "masses",
        [](ParticlesContainer &self)
        {
          return std::vector<Real>(self.masses_gpu.begin(),
                                   self.masses_gpu.end());
        },
        [](ParticlesContainer &self, const std::vector<Real> &value)
        {
          cpu_array<Real> host_val = value;
          self.masses_gpu = host_val;
        },
        "Masses");

    P_cls.def_property(
        "volumes",
        [](ParticlesContainer &self)
        {
          return std::vector<Real>(self.volumes_gpu.begin(),
                                   self.volumes_gpu.end());
        },
        [](ParticlesContainer &self, const std::vector<Real> &value)
        {
          cpu_array<Real> host_val = value;
          self.volumes_gpu = host_val;
        },
        "Volumes (updated every step)");

    P_cls.def_property(
        "volumes_original",
        [](ParticlesContainer &self)
        {
          return std::vector<Real>(self.volumes_original_gpu.begin(),
                                   self.volumes_original_gpu.end());
        },
        [](ParticlesContainer &self, const std::vector<Real> &value)
        {
          cpu_array<Real> host_val = value;
          self.volumes_original_gpu = host_val;
        },
        "Initial volumes");

    P_cls.def_property(
        "colors",
        [](ParticlesContainer &self)
        {
          return std::vector<int>(self.colors_gpu.begin(), self.colors_gpu.end());
        }, // getter
        [](ParticlesContainer &self, const std::vector<int> &value)
        {
          cpu_array<int> host_val = value;
          self.colors_gpu = host_val;
        },
        "Material IDs");

    P_cls.def(py::pickle(
                  [](const ParticlesContainer &particles) { // NOSONAR
                    return pickle_save_particles(particles);
                  },
                  [](py::tuple t) { // NOSONAR
                    return pickle_load_particles(t);
                  }),
              "Pickling of particles container");
  };

} // namespace pyroclastmpm