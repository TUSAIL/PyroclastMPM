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
#include "pyroclastmpm/common/tools.h"
#include "pyroclastmpm/common/types_common.h"

namespace py = pybind11;

namespace pyroclastmpm {

void tools_module(py::module &m) {
  m.def("uniform_random_points_in_volume", &uniform_random_points_in_volume,
        R"(
            Generate a uniform random distribution of points within a volume.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> points = pm.uniform_random_points_in_volume("./stl_file.stl", 1000)

            Parameters
            ----------
            stl_filename: str
                The file name of the name of the STL (.stl).
            num_points: int
                  Number of points to sample.
            
            Returns
            -------
            np.array
                  The sampled points.
             )",
        py::arg("stl_filename"), py::arg("num_points"));

  m.def("grid_points_in_volume", &grid_points_in_volume,
        R"(
            Generate even distribution (grid) of points within a volume.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> points = pm.grid_points_in_volume("./stl_file.stl",  0.01, 2)

            Parameters
            ----------
            stl_filename: str
                The file name of the name of the STL (.stl).
            cell_size: float
                  Grid cell size.
            points_per_cell: int
                  Number of points to sample per cell.
            
            Returns
            -------
            np.array
                  The sampled points.
             )",
        py::arg("stl_filename"), py::arg("cell_size"),
        py::arg("points_per_cell"));

  m.def("grid_points_on_surface", &grid_points_on_surface,
        R"(
            Generate even distribution (grid) of points on a surface.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> points = pm.grid_points_on_surface("./stl_file.stl",  0.01, 2)

            Parameters
            ----------
            stl_filename: str
                The file name of the name of the STL (.stl).
            cell_size: float
                  Grid cell size.
            points_per_cell: int
                  Number of points to sample per cell.
            
            Returns
            -------
            np.array
                  The sampled points.
             )",
        py::arg("stl_filename"), py::arg("cell_size"),
        py::arg("points_per_cell"));

  m.def("get_bounds", &get_bounds,
        R"(
            Get start and end coordinates of an STL file.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> origin,end = pm.get_bounds("./stl_file.stl")

            Parameters
            ----------
            stl_filename: str
                The file name of the name of the STL (.stl).
            
            Returns
            -------
            Tuple[np.array, np.array]
                  A tuple of start and end coordinates
             )",
        py::arg("stl_filename"));

#ifdef CUDA_ENABLED
  m.def("set_device", &set_device,
        R"(
            Sets the GPU device id. Supports only CUDA enabled devices.

            Example usage:
                  >>> import pyroclastmpm as pm
                  >>> pm.set_device(1)

            Parameters
            ----------
            stl_filename: int
                The file name of the name of the STL (.stl).
             )",
        py::arg("device_id"));
#endif
}

} // namespace pyroclastmpm