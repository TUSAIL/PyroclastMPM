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

#pragma once

/**
 * @file tools.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief This file contains tools to generate points from a STL file
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "pyroclastmpm/common/output.h"
#include "pyroclastmpm/common/types_common.h"

// VTK
#include "vtkNew.h"
#include <vtkDataObjectToTable.h>
#include <vtkDelimitedTextWriter.h>
#include <vtkDoubleArray.h>
#include <vtkExtractEnclosedPoints.h>
#include <vtkIntArray.h>
#include <vtkOBJWriter.h>
#include <vtkPointData.h>
#include <vtkPolyDataNormals.h>
#include <vtkPolyDataPointSampler.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>

// STL
#include <random>
#include <tuple>

namespace pyroclastmpm {
std::vector<Vector3r>
uniform_random_points_in_volume(const std::string &stl_filename,
                                const int num_points);

std::vector<Vector3r> grid_points_in_volume(const std::string &stl_filename,
                                            const Real cell_size,
                                            const int point_per_cell);

std::tuple<std::vector<Vector3r>, std::vector<Vector3r>>
grid_points_on_surface(const std::string &stl_filename, const Real cell_size,
                       const int point_per_cell);

std::tuple<Vector3r, Vector3r> get_bounds(const std::string &stl_filename);

#ifdef CUDA_ENABLED
void set_device(int device_id);
#endif

} // namespace pyroclastmpm