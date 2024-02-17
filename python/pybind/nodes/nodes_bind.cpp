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
#include "pyroclastmpm/nodes/nodes.h"

namespace py = pybind11;

namespace pyroclastmpm {

void nodes_module(const py::module &m) {
  py::class_<NodesContainer> N_cls(m, "NodesContainer");
  N_cls.def(py::init<Vectorr, Vectorr, Real>(),
            R"(
    A container for the background grid nodes of the MPM simulation.

    Example usage:
      >>> import pyroclastmpm as pm
      >>> import numpy as np
      >>> origin = np.array([0.0,0.0,0.0])
      >>> end = np.array([1.0,1.0,1.0])
      >>> cell_size = 0.1
      >>> nodes = pm.NodesContainer(origin,end,cell_size)

    Parameters
    ----------
    origin : np.array
        Start coordinates of the grid
    end : np.array
        End coordinates of the grid
    cell_size : float
        Size of the grid cells
    )",
            py::arg("origin"), py::arg("end"), py::arg("cell_size"));

  N_cls.def("set_output_formats", &NodesContainer::set_output_formats,
            R"(
            Set a list of formats that are outputted in a
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

  N_cls.def("give_coords", &NodesContainer::give_node_coords_stl,
            R"(
            A method to get the node coordinates as a flattened array of M nodes in
            D dimensions.

            Returns
            -------
            np.ndarray
            Array of node coordinates of shape 
          
            )");

  N_cls.def_property(
      "moments",
      [](NodesContainer &self) {
        return std::vector<Vectorr>(self.moments_gpu.begin(),
                                    self.moments_gpu.end());
      },
      [](NodesContainer &self, const std::vector<Vectorr> &value) {
        cpu_array<Vectorr> host_val = value;
        self.moments_gpu = host_val;
      },
      "Node moments");
  N_cls.def_property(
      "moments_nt",
      [](NodesContainer &self) {
        return std::vector<Vectorr>(self.moments_nt_gpu.begin(),
                                    self.moments_nt_gpu.end());
      },
      [](NodesContainer &self, const std::vector<Vectorr> &value) {
        cpu_array<Vectorr> host_val = value;
        self.moments_nt_gpu = host_val;
      },
      "Node forward moments (refer to :class:`USL` for more info)");

  N_cls.def_property(
      "forces_external",
      [](NodesContainer &self) {
        return std::vector<Vectorr>(self.forces_external_gpu.begin(),
                                    self.forces_external_gpu.end());
      },
      [](NodesContainer &self, const std::vector<Vectorr> &value) {
        cpu_array<Vectorr> host_val = value;
        self.forces_external_gpu = host_val;
      },
      "External forces on the nodes");
  N_cls.def_property(
      "forces_internal",
      [](NodesContainer &self) {
        return std::vector<Vectorr>(self.forces_internal_gpu.begin(),
                                    self.forces_internal_gpu.end());
      },
      [](NodesContainer &self, const std::vector<Vectorr> &value) {
        cpu_array<Vectorr> host_val = value;
        self.forces_internal_gpu = host_val;
      },
      "Internal forces of the nodes");

  N_cls.def_property(
      "forces_total",
      [](NodesContainer &self) {
        return std::vector<Vectorr>(self.forces_total_gpu.begin(),
                                    self.forces_total_gpu.end());
      },
      [](NodesContainer &self, const std::vector<Vectorr> &value) {
        cpu_array<Vectorr> host_val = value;
        self.forces_total_gpu = host_val;
      },
      "Total forces of the nodes");

  N_cls.def_property(
      "masses",
      [](NodesContainer &self) {
        return std::vector<Real>(self.masses_gpu.begin(),
                                 self.masses_gpu.end());
      },
      [](NodesContainer &self, const std::vector<Real> &value) {
        cpu_array<Real> host_val = value;
        self.masses_gpu = host_val;
      },
      "Masses of the nodes");

  // N_cls.def_readwrite("small_mass_cutoff",
  // &NodesContainer::small_mass_cutoff,
  //                     "Cutoff for small masses default 1.0e-6");
}

} // namespace pyroclastmpm