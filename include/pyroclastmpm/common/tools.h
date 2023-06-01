#pragma once

#include "pyroclastmpm/common/output.h"
#include "pyroclastmpm/common/types_common.h"

// VTK
#include <vtkDataObjectToTable.h>
#include <vtkDelimitedTextWriter.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkOBJWriter.h>
#include <vtkPointData.h>

#include <vtkPolyDataPointSampler.h>

#include "vtkNew.h"
#include <vtkExtractEnclosedPoints.h>
#include <vtkPolyDataNormals.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>

// STL
#include <random>
#include <tuple>

namespace pyroclastmpm {
// const thrust::host_vector<Vector3r> input
std::vector<Vector3r>
uniform_random_points_in_volume(const std::string stl_filename,
                                const int num_points);

std::vector<Vector3r> grid_points_in_volume(const std::string stl_filename,
                                            const Real cell_size,
                                            const int point_per_cell);

std::tuple<std::vector<Vector3r>, std::vector<Vector3r>>
grid_points_on_surface(const std::string stl_filename, const Real cell_size,
                       const int point_per_cell);

std::tuple<Vector3r, Vector3r> get_bounds(const std::string stl_filename);

void set_device(int device_id);

} // namespace pyroclastmpm