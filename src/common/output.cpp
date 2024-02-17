// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
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

/**
 * @file output.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief VTK is primarily used for outputting data
 * @details Functions included here are mainly used to
 * configure a VTK polydata object and write it to file.
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "pyroclastmpm/common/output.h"

namespace pyroclastmpm {

extern const char output_directory_cpu[256];

extern const int global_step_cpu;

extern const Real dt_cpu;

//// @brief Set the vtk points object from the input array
//// @param input input array of points
//// @param polydata VTK polydata object
//// @param mask mask array, should be same size as input (optional)
//// @param use_mask if true, only points where mask is true are added, defaults
/// to false
void set_vtk_points(cpu_array<Vectorr> input,
                    const vtkSmartPointer<vtkPolyData> &polydata,
                    cpu_array<bool> mask, bool use_mask)

{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  for (int id = 0; id < input.size(); id++) {
    // if option is selected to exclude rigid particles, skip them
    if (use_mask && !mask[id]) {
      continue;
    }
#if DIM == 3
    points->InsertNextPoint(input[id][0], input[id][1], input[id][2]);
#elif DIM == 2
    points->InsertNextPoint(input[id][0], input[id][1], 0);
#else
    points->InsertNextPoint(input[id][0], 0, 0);
#endif
  }
  polydata->SetPoints(points);
}

/// @brief Set the values for vtk pointdata object from the input array
/// @tparam T data type (float,double, int, Matrixr, etc.)
/// @param input Input array of values
/// @param polydata VTK polydata object
/// @param pointdata_name Name variables being set (e.g. "velocity")
/// @param mask Mask array, should be same size as input (optional)
/// @param use_mask If true, only points where mask is true are added, defaults
/// to false
template <typename T>
void set_vtk_pointdata(cpu_array<T> input,
                       const vtkSmartPointer<vtkPolyData> &polydata,
                       const std::string &pointdata_name, cpu_array<bool> mask,
                       bool use_mask) {

  vtkSmartPointer<vtkDoubleArray> pointdata =
      vtkSmartPointer<vtkDoubleArray>::New();

  if constexpr (std::is_same_v<T, Matrixr> || std::is_same_v<T, Vectorr> ||
                std::is_same_v<T, Matrix3r>) {
    pointdata->SetNumberOfComponents(static_cast<int>(input[0].size()));

  } else if constexpr (std::is_same_v<T, Vectori>) {
    pointdata->SetNumberOfComponents(static_cast<int>(input[0].size()));
  } else {

    pointdata->SetNumberOfComponents(1);
  }

  pointdata->SetName(pointdata_name.c_str());

  for (int id = 0; id < input.size(); id++) {
    if (use_mask && !mask[id]) {
      continue;
    }
    if constexpr (std::is_same_v<T, Matrixr> || std::is_same_v<T, Vectorr> ||
                  std::is_same_v<T, Matrix3r>) {
      const Real *data = input[0].data();
      pointdata->InsertNextTuple(data + id * input[0].size());
    } else if constexpr (std::is_same_v<T, Vectori>) {
      // TODO: fix this (print node_ids as int)
      //  const Real *data = input[0].data();
      //  pointdata->InsertNextTuple(data + id * input[0].size());
    } else {
      pointdata->InsertNextValue(input[id]);
    }
  }

  polydata->GetPointData()->AddArray(pointdata);
}

///@brief write output to different formats
///@param polydata Output vtk polydata
///@param filestem Filestem for output file (e.g. "particles")
///@param output_type Output type  ("vtk", "gtfl", "obj", "csv")
void write_vtk_polydata(vtkSmartPointer<vtkPolyData> polydata,
                        const std::string &filestem,
                        const std::string_view &output_type)

{

  // Add some Metadata
  vtkNew<vtkDoubleArray> current_time;
  current_time->SetName("Time");

  current_time->SetNumberOfComponents(1);
  current_time->InsertNextValue(static_cast<Real>(global_step_cpu) * dt_cpu);
  polydata->GetFieldData()->AddArray(current_time);

  std::string output_directory = output_directory_cpu;
  std::string filename =
      output_directory + "/" + filestem + std::to_string(global_step_cpu);

  if (output_type == "vtk") {
    filename += ".vtp";

    vtkSmartPointer<vtkXMLPolyDataWriter> writer =
        vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polydata);
    writer->Write();
  } else if (output_type == "gtfl") {
    filename += ".gtfl";
    vtkSmartPointer<vtkMultiBlockDataSet> multiblock =
        vtkSmartPointer<vtkMultiBlockDataSet>::New();
    multiblock->SetNumberOfBlocks(1);
    multiblock->SetBlock(0, polydata);
    vtkSmartPointer<vtkGLTFWriter> writer =
        vtkSmartPointer<vtkGLTFWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(multiblock);
    writer->Write();
  } else if (output_type == "obj") {
    filename += ".obj";
    vtkOBJWriter *writer = vtkOBJWriter::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polydata);
    writer->Write();
  } else if (output_type == "csv") {
    vtkSmartPointer<vtkDataObjectToTable> toTable =
        vtkSmartPointer<vtkDataObjectToTable>::New();
    toTable->SetInputData(polydata);

    toTable->SetFieldType(vtkDataObjectToTable::POINT_DATA);
    toTable->Update();
    vtkTable *table = toTable->GetOutput();

    filename += ".csv";
    vtkSmartPointer<vtkDelimitedTextWriter> writer =
        vtkSmartPointer<vtkDelimitedTextWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(table);
    writer->Write();
  }
}

// Explicit template instantiation
template void set_vtk_pointdata<bool>(
    cpu_array<bool> input, const vtkSmartPointer<vtkPolyData> &polydata,
    const std::string &pointdata_name, cpu_array<bool> mask, bool use_mask);

template void set_vtk_pointdata<uint8_t>(
    cpu_array<uint8_t> input, const vtkSmartPointer<vtkPolyData> &polydata,
    const std::string &pointdata_name, cpu_array<bool> mask, bool use_mask);

template void set_vtk_pointdata<int>(
    cpu_array<int> input, const vtkSmartPointer<vtkPolyData> &polydata,
    const std::string &pointdata_name, cpu_array<bool> mask, bool use_mask);

template void set_vtk_pointdata<Real>(
    cpu_array<Real> input, const vtkSmartPointer<vtkPolyData> &polydata,
    const std::string &pointdata_name, cpu_array<bool> mask, bool use_mask);

template void set_vtk_pointdata<Vectorr>(
    cpu_array<Vectorr> input, const vtkSmartPointer<vtkPolyData> &polydata,
    const std::string &pointdata_name, cpu_array<bool> mask, bool use_mask);

template void set_vtk_pointdata<Vectori>(
    cpu_array<Vectori> input, const vtkSmartPointer<vtkPolyData> &polydata,
    const std::string &pointdata_name, cpu_array<bool> mask, bool use_mask);

#if DIM != 1
template void set_vtk_pointdata<Matrixr>(
    cpu_array<Matrixr> input, const vtkSmartPointer<vtkPolyData> &polydata,
    const std::string &pointdata_name, cpu_array<bool> mask, bool use_mask);

#endif

#if DIM != 3
template void set_vtk_pointdata<Matrix3r>(
    cpu_array<Matrix3r> input, const vtkSmartPointer<vtkPolyData> &polydata,
    const std::string &pointdata_name, cpu_array<bool> mask, bool use_mask);
#endif

} // namespace pyroclastmpm