#pragma once

// VTK
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
void set_vtk_points(cpu_array<Vectorr> input,
                    vtkSmartPointer<vtkPolyData> &polydata,
                    cpu_array<bool> mask = {}, bool use_mask = false);

template <typename T>
void set_vtk_pointdata(cpu_array<T> input,
                       vtkSmartPointer<vtkPolyData> &polydata,
                       const std::string pointdata_name,
                       cpu_array<bool> mask = {}, bool use_mask = false);

void write_vtk_polydata(vtkSmartPointer<vtkPolyData> polydata,
                        const std::string filename,
                        OutputType output_type = VTK);

} // namespace pyroclastmpm