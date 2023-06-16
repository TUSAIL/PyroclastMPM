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
 * @file output.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief This header file that contains functions for outputting
 * data
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <filesystem>
#include <iostream>
#include <type_traits>
#include <vtkDataObjectToTable.h>
#include <vtkDelimitedTextWriter.h>
#include <vtkDoubleArray.h>
#include <vtkGLTFWriter.h>
#include <vtkIntArray.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkOBJWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>
#include <vtkXMLPolyDataWriter.h>

#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm {

/**
 * @brief Set the vtk points locations
 *
 * @param input input array of points
 * @param polydata polydata vtk object to set the points
 * @param mask mask to use for which points to set (input size and mask size
 * should be the same)
 * @param use_mask if true, use the mask to set the points, otherwise set all
 */
void set_vtk_points(cpu_array<Vectorr> input,
                    const vtkSmartPointer<vtkPolyData> &polydata,
                    cpu_array<bool> mask = {}, bool use_mask = false);

/**
 * @brief Set the vtk pointdata to input polydata
 *
 * @tparam T data type (float,double, int, Matrixr, etc.)
 * @param input input array of data
 * @param polydata polydata vtk object to set the pointdata
 * @param pointdata_name name of the pointdata
 * @param mask mask to use for which points to set (input size and mask size
 * should be the same)
 * @param use_mask if true, use the mask to set the points, otherwise set all
 */
template <typename T>
void set_vtk_pointdata(cpu_array<T> input,
                       vtkSmartPointer<vtkPolyData> &polydata,
                       const std::string pointdata_name,
                       cpu_array<bool> mask = {}, bool use_mask = false);

/**
 * @brief Set the vtk pointdata to input polydata
 *
 * @param polydata polydata vtk object to set the pointdata
 * @param filename filename to write to
 * @param output_type output type (VTK, OBJ, GLTF,etc.)
 */
void write_vtk_polydata(vtkSmartPointer<vtkPolyData> polydata,
                        const std::string filename,
                        OutputType output_type = VTK);

} // namespace pyroclastmpm