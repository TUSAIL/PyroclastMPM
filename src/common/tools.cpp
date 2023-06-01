

#include "pyroclastmpm/common/tools.h"

namespace pyroclastmpm {

/**
 * @brief a function to sample a random number of points in a volume
 *
 * @param stl_filename string containing the path to the stl file
 * @param num_points number of points to sample
 * @return std::vector<Vector3r>
 */
std::vector<Vector3r>
uniform_random_points_in_volume(const std::string stl_filename,
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

  std::mt19937 mt(4355412); // Standard mersenne_twister_engine
  double bounds[6];
  geometry->GetBounds(bounds);
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
    double point[3];
    point[0] = distributionX(mt);
    point[1] = distributionY(mt);
    point[2] = distributionZ(mt);
    points->SetPoint(i, point);
  }

  vtkNew<vtkExtractEnclosedPoints> extract;
  extract->SetSurfaceData(geometry);
  extract->SetInputData(pointsPolyData);
  extract->SetTolerance(.001);
  extract->CheckSurfaceOn();
  extract->Update();

  std::vector<Vector3r> positions_cpu;
  positions_cpu.resize(num_points);

  for (int id = 0; id < num_points; id++) {
    double *p = new double;
    p = extract->GetOutput()->GetPoint(id);
    positions_cpu[id][0] = p[0];
    positions_cpu[id][1] = p[1];
    positions_cpu[id][2] = p[2];
  }
  return positions_cpu;
}

std::vector<Vector3r> grid_points_in_volume(const std::string stl_filename,
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

  std::mt19937 mt(4355412); // Standard mersenne_twister_engine
  double bounds[6];         // xmin,xmax,ymin,ymax,zmin,zmax
  geometry->GetBounds(bounds);
  std::cout << "Bounds: " << bounds[0] << ", " << bounds[1] << " " << bounds[2]
            << ", " << bounds[3] << " " << bounds[4] << ", " << bounds[5]
            << std::endl;

  int grid_sizes[3];

  Real gap = cell_size / point_per_cell;

  grid_sizes[0] = abs(bounds[1] - bounds[0]) / gap + 1.;
  grid_sizes[1] = abs(bounds[3] - bounds[2]) / gap + 1.;
  grid_sizes[2] = abs(bounds[5] - bounds[4]) / gap + 1.;

  vtkNew<vtkPolyData> pointsPolyData;
  vtkNew<vtkPoints> points;
  pointsPolyData->SetPoints(points);

  points->SetNumberOfPoints(grid_sizes[0] * grid_sizes[1] * grid_sizes[2]);
  for (auto xi = 0; xi < grid_sizes[0]; ++xi) {
    for (auto yi = 0; yi < grid_sizes[1]; ++yi) {
      for (auto zi = 0; zi < grid_sizes[2]; ++zi) {
        double point[3];
        point[0] = bounds[0] + 0.5 * gap + xi * gap;
        point[1] = bounds[2] + 0.5 * gap + yi * gap;
        point[2] = bounds[4] + 0.5 * gap + zi * gap;

        const unsigned int index =
            xi + yi * grid_sizes[0] + zi * grid_sizes[0] * grid_sizes[1];

        points->SetPoint(index, point);
      }
    }
  }

  vtkNew<vtkExtractEnclosedPoints> extract;
  extract->SetSurfaceData(geometry);
  extract->SetInputData(pointsPolyData);
  extract->SetTolerance(.00000000001);
  // extract->CheckSurfaceOn();
  extract->Update();

  int num_points = extract->GetOutput()->GetNumberOfPoints();
  std::vector<Vector3r> positions_cpu;
  positions_cpu.resize(num_points);

  for (int id = 0; id < num_points; id++) {
    double *p = new double;
    p = extract->GetOutput()->GetPoint(id);
    positions_cpu[id][0] = p[0];
    positions_cpu[id][1] = p[1];
    positions_cpu[id][2] = p[2];
  }
  return positions_cpu;
  // write_vtk_polydata(extract->GetOutput(), "distribute");
  // write_vtk_polydata(pointsPolyData, "distribute");
}

std::tuple<std::vector<Vector3r>, std::vector<Vector3r>>
grid_points_on_surface(const std::string stl_filename, const Real cell_size,
                       const int point_per_cell) {
  vtkNew<vtkSTLReader> reader;
  reader->SetFileName(stl_filename.c_str());
  reader->Update();

  vtkNew<vtkPolyDataNormals> normals;
  normals->SetInputConnection(reader->GetOutputPort());
  normals->FlipNormalsOn();

  vtkNew<vtkPolyDataPointSampler> sampler;

  Real gap = cell_size / point_per_cell;

  sampler->SetInputConnection(normals->GetOutputPort());
  sampler->SetPointGenerationModeToRegular();
  sampler->InterpolatePointDataOn();
  sampler->SetDistance(gap);
  sampler->Update();

  // geometry = vtkSmartPointer<vtkPolyData>::New();

  // geometry = sampler->GetOutput();

  // write_vtk_polydata(sampler->GetOutput(), "distribute_surface");
  int num_points = sampler->GetOutput()->GetNumberOfPoints();
  std::vector<Vector3r> positions_cpu;
  positions_cpu.resize(num_points);

  for (int id = 0; id < num_points; id++) {
    double *p = new double;
    p = sampler->GetOutput()->GetPoint(id);
    positions_cpu[id][0] = p[0];
    positions_cpu[id][1] = p[1];
    positions_cpu[id][2] = p[2];
  }

  // TODO make normals

  return std::make_tuple(positions_cpu, positions_cpu);
}

std::tuple<Vector3r, Vector3r> get_bounds(const std::string stl_filename) {
  vtkNew<vtkSTLReader> reader;
  reader->SetFileName(stl_filename.c_str());
  reader->Update();

  double bounds[6]; // xmin,xmax,ymin,ymax,zmin,zmax
  reader->GetOutput()->GetBounds(bounds);

  Vector3r bound_start = {(float)bounds[0], (float)bounds[2], (float)bounds[4]};

  Vector3r bound_end = {(float)bounds[1], (float)bounds[3], (float)bounds[5]};

  return std::make_tuple(bound_start, bound_end);
}

void set_device(int device_id) {

#ifdef CUDA_ENABLED
  cudaSetDevice(device_id);

  gpuErrchk(cudaDeviceSynchronize());

#endif
}

} // namespace pyroclastmpm