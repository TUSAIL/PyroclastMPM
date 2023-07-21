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

// PYBIND
#include "pybind11/eigen.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// MODULE
#include "pyroclastmpm/common/global_settings.h"
#include "pyroclastmpm/common/types_common.h"

namespace py = pybind11;

namespace pyroclastmpm {

void global_settings_module(py::module &m) {
  m.def("set_global_shapefunction", &set_global_shapefunction,
        R"(
            Sets the global shape function for the MPM simulation. 

            Example usage:
                  >>> import pyroclastmpm.MPM3D as pm
                  >>> pm.set_global_shapefunction("linear")

            Parameters
            ----------
            shapefunction: str
                Type of shape function to use  "linear" or "cubic", by default "linear".

             )",
        py::arg("shapefunction") = "linear");

  m.def("set_global_timestep", &set_global_dt,
        R"(
            Sets the global time step for the MPM simulation. 

            Example usage:
                  >>> import pyroclastmpm.MPM3D as pm
                  >>> pm.set_global_timestep(0.01)

            Parameters
            ----------
            timestep: float
                Input time step.

             )",
        py::arg("timestep"));

  m.def("set_global_step", &set_global_step,
        R"(
            Sets the global step for the MPM simulation. 

            Example usage:
                  >>> import pyroclastmpm.MPM3D as pm
                  >>> pm.set_global_step(10)

            Parameters
            ----------
            step: int
                input step.
             )",
        py::arg("step"));

  m.def("set_global_output_directory", &set_global_output_directory,
        R"(
            Sets the output directory for the MPM simulation where
            the output files will be saved.

            Example usage:
                  >>> import pyroclastmpm.MPM3D as pm
                  >>> pm.set_global_output_directory("./output/")

            Parameters
            ----------
            directory: str
                Folder path to save the output files.
             )",
        py::arg("directory"));

  m.def("set_globals", &set_globals,
        R"(
            Sets the global variables for the MPM simulation.

            Example usage:
                  >>> import pyroclastmpm.MPM3D as pm
                  >>> pm.set_global_step(0.001, 4, "linear", "./output/")

            Parameters
            ----------
            timestep: float
                Input time step.
            particles_per_cell: int
                Number of particles per cell. Recommended to use at least
                2 particles per cell for 1D, 4 for 2D and 8 for 3D.
            shapefunction: str
                  Type of shape function to use  "linear" or "cubic".
            directory: str
                  Folder path to save the output files.
            )",
        py::arg("timestep"), py::arg("particles_per_cell"),
        py::arg("shapefunction") = "linear",
        py::arg("directory") = "./output/");
}

} // namespace pyroclastmpm