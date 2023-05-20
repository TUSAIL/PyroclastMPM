
#include "pyroclastmpm/common/output.cuh"

namespace pyroclastmpm
{

  extern char output_directory_cpu[256];

  extern OutputType output_type_cpu;

  extern int global_step_cpu;

  extern Real dt_cpu;

  void set_vtk_points(cpu_array<Vectorr> input,
                      vtkSmartPointer<vtkPolyData> &polydata)

  {
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    for (int id = 0; id < input.size(); id++)
    {
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

  template <typename T>
  void set_vtk_pointdata(cpu_array<T> input,
                         vtkSmartPointer<vtkPolyData> &polydata,
                         const std::string pointdata_name)
  {

    vtkSmartPointer<vtkDoubleArray> pointdata =
        vtkSmartPointer<vtkDoubleArray>::New();

    if constexpr (std::is_same_v<T, Matrixr> || std::is_same_v<T, Vectorr> || std::is_same_v<T, Matrix3r>)
    {
      // use type specific operations...
      pointdata->SetNumberOfComponents(input[0].size());
    }
    else
    {

      pointdata->SetNumberOfComponents(1);
    }

    pointdata->SetName(pointdata_name.c_str());

    for (int id = 0; id < input.size(); id++)
    {
      if constexpr (std::is_same_v<T, Matrixr> || std::is_same_v<T, Vectorr> || std::is_same_v<T, Matrix3r>)
      {
        Real *data = input[0].data();
        pointdata->InsertNextTuple(data + id * input[0].size());
      }
      else
      {
        pointdata->InsertNextValue(input[id]);
      }
    }

    polydata->GetPointData()->AddArray(pointdata);
  }

  template void set_vtk_pointdata<uint8_t>(cpu_array<uint8_t> input,
                                           vtkSmartPointer<vtkPolyData> &polydata,
                                           const std::string pointdata_name);

  template void set_vtk_pointdata<int>(cpu_array<int> input,
                                       vtkSmartPointer<vtkPolyData> &polydata,
                                       const std::string pointdata_name);

  template void set_vtk_pointdata<Real>(cpu_array<Real> input,
                                        vtkSmartPointer<vtkPolyData> &polydata,
                                        const std::string pointdata_name);

  template void set_vtk_pointdata<Vectorr>(cpu_array<Vectorr> input,
                                           vtkSmartPointer<vtkPolyData> &polydata,
                                           const std::string pointdata_name);

// Since Matri3r is a typedef of Matrixr is same as Vector (explicitly initiated )
#if DIM != 1
  template void set_vtk_pointdata<Matrixr>(cpu_array<Matrixr> input,
                                           vtkSmartPointer<vtkPolyData> &polydata,
                                           const std::string pointdata_name);
#endif

// Since Matri3r is the same as Matrixr, (explicitly initiated )
#if DIM != 3
  template void set_vtk_pointdata<Matrix3r>(cpu_array<Matrix3r> input,
                                            vtkSmartPointer<vtkPolyData> &polydata,
                                            const std::string pointdata_name);
#endif

  void write_vtk_polydata(vtkSmartPointer<vtkPolyData> polydata,
                          const std::string filestem, OutputType output_type)

  {

    // Add some Metadata
    vtkNew<vtkDoubleArray> current_time;
    current_time->SetName("Time");

    current_time->SetNumberOfComponents(1);
    current_time->InsertNextValue(global_step_cpu * dt_cpu);
    polydata->GetFieldData()->AddArray(current_time);
    // vtkNew<vtkDoubleArray> current_step;
    // vtkNew<vtkDoubleArray> dt;

    std::string output_directory = output_directory_cpu;
    std::string filename =
        output_directory + "/" + filestem + std::to_string(global_step_cpu);

    if (output_type == VTK)
    {
      filename += ".vtp";

      vtkSmartPointer<vtkXMLPolyDataWriter> writer =
          vtkSmartPointer<vtkXMLPolyDataWriter>::New();
      writer->SetFileName(filename.c_str());
      writer->SetInputData(polydata);
      writer->Write();
    }
    else if (output_type == OBJ)
    {
      filename += ".obj";
      vtkOBJWriter *writer = vtkOBJWriter::New();
      writer->SetFileName(filename.c_str());
      writer->SetInputData(polydata);
      writer->Write();
    }
    else if (output_type == CSV)
    {
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
    else if (output_type == HDF5)
    {
      // output to hdf5 file using xdmf
    }
  }

} // namespace pyroclastmpm