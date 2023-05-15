#pragma once

// VTK
#include <vtkDataObjectToTable.h>
#include <vtkDelimitedTextWriter.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkOBJWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>
#include <vtkXMLPolyDataWriter.h>
#include <filesystem>
#include <iostream>
#include <type_traits>

#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm
{
    void set_vtk_points(cpu_array<Vectorr> input,
                        vtkSmartPointer<vtkPolyData> &polydata);

    template <typename T>
    void set_vtk_pointdata(cpu_array<T> input,
                           vtkSmartPointer<vtkPolyData> &polydata,
                           const std::string pointdata_name);

    void write_vtk_polydata(vtkSmartPointer<vtkPolyData> polydata,
                            const std::string filename, OutputType output_type= VTK);

} // namespace pyroclastmpm