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
 * @brief Set the vtk points locations to input polydata
 * @details This function is used to populate a polydata
 * with points
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include <vtkPolyData.h>
 *        #include "pyroclastmpm/common/output.h"
 *
 *        vtkSmartPointer<vtkPolyData> polydata =
 * vtkSmartPointer<vtkPolyData>::New();
 *
 *        std::vector<Vectorr> points = {Vectorr({0., 0.25, 0.}),...};
 *
 *        set_vtk_points(positions_cpu, polydata);
 *
 *
 * \endverbatim
 *
 *
 * @param input Input array of points
 * @param polydata Polydata vtk object to set the points
 * @param mask <ask to use for which points to set (input size and mask size
 * should be the same)
 * @param use_mask If true, use the mask to set the points, otherwise set all
 */
void set_vtk_points(cpu_array<Vectorr> input,
                    const vtkSmartPointer<vtkPolyData> &polydata,
                    cpu_array<bool> mask = {}, bool use_mask = false);

/**
 * @brief Set the vtk pointdata to input polydata
 * @details This function is used to populate a polydata with scalar,
 * vector, or tensor pointdata.
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include <vtkPolyData.h>
 *        #include "pyroclastmpm/common/output.h"
 *
 *        ...
 *        set_vtk_points(positions_cpu, polydata);
 *
 *        // Uses no mask by default
 *        set_vtk_pointdata<Vectorr>(moments_cpu, polydata, "Moments");
 *
 *        // With mask {true, false, true, ...}
 *        set_vtk_pointdata<Vectorr>(moments_cpu, polydata, "Moments", mask,
 * true);
 *
 *
 *
 * \endverbatim
 *
 * @tparam T data type (float,double, int, Matrixr, etc.)
 * @param input Input array of data
 * @param polydata Polydata vtk object to set the pointdata
 * @param pointdata_name Name of the pointdata
 * @param mask Mask to use for which points to set (input size and mask size
 * should be the same)
 * @param use_mask If true, use the mask to set the points, otherwise set all
 */
template <typename T>
void set_vtk_pointdata(cpu_array<T> input,
                       const vtkSmartPointer<vtkPolyData> &polydata,
                       const std::string &pointdata_name,
                       cpu_array<bool> mask = {}, bool use_mask = false);

/**
 * @brief write an output file
 * @details Outputs a vtkpolydata object to a file
 * can be used to write to vtk, obj, and csv formats.
 * It is very important that output directory is set
 * before calling this function.
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include <vtkPolyData.h>
 *        #include "pyroclastmpm/common/output.h"
 *
 *        ...
 *        set_vtk_points(positions_cpu, polydata);
 *
 *        // Uses no mask by default
 *        set_vtk_pointdata<Vectorr>(moments_cpu, polydata, "Moments");
 *
 *       ...
 *       set_global_output_directory("./output/") // NB: this is very important
 *
 *       write_vtk_polydata(polydata, "data.vtk", "csv");
 *
 * \endverbatim
 *
 * @param polydata Polydata vtk object to set the pointdata
 * @param filename Filename to write to
 * @param output_type Output type ("vtk", "obj", "csv" etc.)
 */
void write_vtk_polydata(vtkSmartPointer<vtkPolyData> polydata,
                        const std::string &filename,
                        const std::string_view &output_type = "vtk");

} // namespace pyroclastmpm