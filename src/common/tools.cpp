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
 * @file tools.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Contains tools to generate points in a volume or on a surface (STL)
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "pyroclastmpm/common/tools.h"

namespace pyroclastmpm {

/// @brief Samples points randomly in a volume
/// @param stl_filename String containing the path to the STL file
/// @param num_points Number of points to sample
/// @return std::vector<Vector3r> output points
std::vector<Vector3r>
uniform_random_points_in_volume(const std::string &stl_filename,
                                const int num_points) {
  vtkNew<vtkSTLReader> reader;
  reader->SetFileName(stl_filename.c_str());
  reader->Update();

  vtkNew<vtkPolyDataNormals> normals;
  normals->SetInputConnection(reader->GetOutputPort());
  normals->FlipNormalsOn();
  normals->Update();

  vtkSmartPointer<vtkPolyData> geometry = vtkSmartPointer<vtkPolyData>::New();

  geometry = normals->GetOutput();

  std::mt19937 mt(4355412);

  std::array<Real, 6> bounds;

  std::copy(bounds.begin(), bounds.end(), geometry->GetBounds());

  std::cout << "Bounds: " << bounds[0] << ", " << bounds[1] << " " << bounds[2]
            << ", " << bounds[3] << " " << bounds[4] << ", " << bounds[5]
            << std::endl;
  // Generate random points within the bounding box of the polydata
  std::uniform_real_distribution<double> distributionX(bounds[0], bounds[1]);
  std::uniform_real_distribution<double> distributionY(bounds[2], bounds[3]);
  std::uniform_real_distribution<double> distributionZ(bounds[4], bounds[5]);
  vtkNew<vtkPolyData> pointsPolyData;
  vtkNew<vtkPoints> points;
  pointsPolyData->SetPoints(points);

  points->SetNumberOfPoints(num_points);
  for (auto i = 0; i < num_points; ++i) {
    std::array<Real, 3> point;
    point[0] = static_cast<Real>(distributionX(mt));
    point[1] = static_cast<Real>(distributionY(mt));
    point[2] = static_cast<Real>(distributionZ(mt));
    points->SetPoint(i, point[0], point[1], point[2]);
  }

  vtkNew<vtkExtractEnclosedPoints> extract;
  extract->SetSurfaceData(geometry);
  extract->SetInputData(pointsPolyData);
  extract->SetTolerance(.001);
  extract->CheckSurfaceOn();
  extract->Update();

  std::vector<Vector3r> positions_cpu;
  positions_cpu.resize(num_points);

  std::array<Real, 3> point;
  for (int id = 0; id < num_points; id++) {
    std::copy(point.begin(), point.end(), extract->GetOutput()->GetPoint(id));
    positions_cpu[id][0] = point[0];
    positions_cpu[id][1] = point[1];
    positions_cpu[id][2] = point[2];
  }

  return positions_cpu;
}

/// @brief Samples points as a grid within a volume
/// @param stl_filename String containing the path to the STL file
/// @param cell_size Cell spacing of each point in the grid
/// @param point_per_cell Number of points per cell
/// @return std::vector<Vector3r> Output points
std::vector<Vector3r> grid_points_in_volume(const std::string &stl_filename,
                                            const Real cell_size,
                                            const int point_per_cell) {
  vtkNew<vtkSTLReader> reader;
  reader->SetFileName(stl_filename.c_str());
  reader->Update();

  vtkNew<vtkPolyDataNormals> normals;
  normals->SetInputConnection(reader->GetOutputPort());
  normals->FlipNormalsOn();
  normals->Update();

  vtkSmartPointer<vtkPolyData> geometry = vtkSmartPointer<vtkPolyData>::New();

  geometry = normals->GetOutput();

  std::mt19937 mt(4355412);

  // xmin,xmax,ymin,ymax,zmin,zmax
  std::array<Real, 6> bounds;

  std::copy(bounds.begin(), bounds.end(), geometry->GetBounds());

  std::cout << "Bounds: " << bounds[0] << ", " << bounds[1] << " " << bounds[2]
            << ", " << bounds[3] << " " << bounds[4] << ", " << bounds[5]
            << std::endl;

  std::array<int, 3> grid_sizes;

  Real gap = cell_size / static_cast<Real>(point_per_cell);

  grid_sizes[0] = (int)floor(abs(bounds[1] - bounds[0]) / gap + 1.0);
  grid_sizes[1] = (int)floor(abs(bounds[3] - bounds[2]) / gap + 1.0);
  grid_sizes[2] = (int)floor(abs(bounds[5] - bounds[4]) / gap + 1.0);

  vtkNew<vtkPolyData> pointsPolyData;
  vtkNew<vtkPoints> points;
  pointsPolyData->SetPoints(points);

  points->SetNumberOfPoints(grid_sizes[0] * grid_sizes[1] * grid_sizes[2]);
  for (auto xi = 0; xi < grid_sizes[0]; ++xi) {
    for (auto yi = 0; yi < grid_sizes[1]; ++yi) {
      for (auto zi = 0; zi < grid_sizes[2]; ++zi) {

        std::array<Real, 3> point;
        point[0] = bounds[0] + ((Real)0.5) * gap + ((Real)xi) * gap;
        point[1] = bounds[2] + ((Real)0.5) * gap + ((Real)yi) * gap;
        point[2] = bounds[4] + ((Real)0.5) * gap + ((Real)zi) * gap;

        const unsigned int index =
            xi + yi * grid_sizes[0] + zi * grid_sizes[0] * grid_sizes[1];

        points->SetPoint(index, point[0], point[1], point[2]);
      }
    }
  }

  vtkNew<vtkExtractEnclosedPoints> extract;
  extract->SetSurfaceData(geometry);
  extract->SetInputData(pointsPolyData);
  extract->SetTolerance(.00000000001);
  extract->Update();

  auto num_points = (int)extract->GetOutput()->GetNumberOfPoints();

  std::vector<Vector3r> positions_cpu;
  positions_cpu.resize(num_points);

  std::array<Real, 3> point;
  for (int id = 0; id < num_points; id++) {
    std::copy(point.begin(), point.end(), extract->GetOutput()->GetPoint(id));
    positions_cpu[id][0] = point[0];
    positions_cpu[id][1] = point[1];
    positions_cpu[id][2] = point[2];
  }
  return positions_cpu;
}

/// @brief Samples points as a grid on the surface of an STL file
/// @param stl_filename String containing the path to the STL file
/// @param cell_size Cell spacing of each point in the grid
/// @param point_per_cell Number of points per cell
/// @return std::vector<Vector3r> Output points
std::tuple<std::vector<Vector3r>, std::vector<Vector3r>>
grid_points_on_surface(const std::string &stl_filename, const Real cell_size,
                       const int point_per_cell) {
  vtkNew<vtkSTLReader> reader;
  reader->SetFileName(stl_filename.c_str());
  reader->Update();

  vtkNew<vtkPolyDataNormals> normals;
  normals->SetInputConnection(reader->GetOutputPort());
  normals->FlipNormalsOn();

  vtkNew<vtkPolyDataPointSampler> sampler;

  Real gap = cell_size / static_cast<Real>(point_per_cell);

  sampler->SetInputConnection(normals->GetOutputPort());
  sampler->SetPointGenerationModeToRegular();
  sampler->InterpolatePointDataOn();
  sampler->SetDistance(gap);
  sampler->Update();

  auto num_points = (int)sampler->GetOutput()->GetNumberOfPoints();
  std::vector<Vector3r> positions_cpu;
  positions_cpu.resize(num_points);

  std::array<Real, 3> point;
  for (int id = 0; id < num_points; id++) {
    std::copy(point.begin(), point.end(), sampler->GetOutput()->GetPoint(id));
    ;
    positions_cpu[id][0] = point[0];
    positions_cpu[id][1] = point[1];
    positions_cpu[id][2] = point[2];
  }

  // TODO make normals

  return std::make_tuple(positions_cpu, positions_cpu);
}

/// @brief Get the start and end position of the bounding box of an STL file
/// @param stl_filename  String containing the path to the STL file
/// @return std::tuple<Vector3r, Vector3r> Tuple containing the start and end
std::tuple<Vector3r, Vector3r> get_bounds(const std::string &stl_filename) {
  vtkNew<vtkSTLReader> reader;
  reader->SetFileName(stl_filename.c_str());
  reader->Update();

  // xmin,xmax,ymin,ymax,zmin,zmax
  std::array<Real, 6> bounds;

  std::copy(bounds.begin(), bounds.end(), reader->GetOutput()->GetBounds());

  Vector3r bound_start = {bounds[0], bounds[2], bounds[4]};

  Vector3r bound_end = {bounds[1], bounds[3], bounds[5]};

  return std::make_tuple(bound_start, bound_end);
}

#ifdef CUDA_ENABLED
/// @brief Set the GPU device id to run on
void set_device(int device_id) {

  cudaSetDevice(device_id);

  gpuErrchk(cudaDeviceSynchronize());
}

#endif

} // namespace pyroclastmpm