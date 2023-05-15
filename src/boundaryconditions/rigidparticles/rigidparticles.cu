#include "pyroclastmpm/boundaryconditions/rigidparticles/rigidparticles.cuh"

namespace pyroclastmpm
{

    extern int global_step_cpu;

    extern Real dt_cpu;

    RigidParticles::RigidParticles(const cpu_array<Vectorr> _positions,
                                   const cpu_array<int> _frames,
                                   const cpu_array<Vectorr> _locations,
                                   const cpu_array<Vectorr> _rotations,
                                   const cpu_array<OutputType> _output_formats
                                   )
    {
        output_formats = _output_formats;
        num_particles = _positions.size();

        set_default_device<Vectorr>(num_particles, _positions, positions_gpu, Vectorr::Zero());
        set_default_device<Vectorr>(num_particles, {}, velocities_gpu, Vectorr::Zero());

        launch_config.tpb = dim3(int((num_particles) / BLOCKSIZE) + 1, 1, 1);
        launch_config.bpg = dim3(BLOCKSIZE, 1, 1);

        num_frames = _frames.size();

        frames_cpu = _frames;

        locations_cpu = _locations;

        rotations_cpu = _rotations;

        // initial Center of mass
        COM = Vectorr::Zero();
        for (int pid = 0; pid < num_particles; pid++)
        {
            COM += _positions[pid];
        }
        COM /= num_particles;

        // initial translational velocity
        translational_velocity = Vectorr::Zero();

        // initial euler angles
        ROT = Vectorr::Zero();

        // initial rotation matrix
        rotation_matrix = Matrixr::Zero();
    }

    void RigidParticles::partition()
    {
        spatial.calculate_hash(positions_gpu);

        spatial.sort_hashes();

        spatial.bin_particles();
    };

    void RigidParticles::initialize(NodesContainer &nodes_ref,
                                    ParticlesContainer &particles_ref)
    {
        spatial = SpatialPartition(nodes_ref.node_start, nodes_ref.node_end,
                                   nodes_ref.node_spacing, num_particles);

        set_default_device<Vectorr>(nodes_ref.num_nodes_total, {}, normals_gpu, Vectorr::Zero());
        set_default_device<bool>(nodes_ref.num_nodes_total, {}, is_overlapping_gpu, false);
        set_default_device<int>(nodes_ref.num_nodes_total, {}, closest_rigid_particle_gpu,
                                -1);
    }

    void RigidParticles::calculate_non_rigid_grid_normals(
        NodesContainer &nodes_ref,
        ParticlesContainer &particles_ref)
    {
        KERNELS_CALC_NON_RIGID_GRID_NORMALS<<<nodes_ref.launch_config.tpb,
                                              nodes_ref.launch_config.bpg>>>(
            thrust::raw_pointer_cast(normals_gpu.data()),
            thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
            thrust::raw_pointer_cast(particles_ref.dpsi_gpu.data()),
            thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
            thrust::raw_pointer_cast(particles_ref.spatial.cell_start_gpu.data()),
            thrust::raw_pointer_cast(particles_ref.spatial.cell_end_gpu.data()),
            thrust::raw_pointer_cast(particles_ref.spatial.sorted_index_gpu.data()),
            particles_ref.spatial.num_cells, particles_ref.spatial.num_cells_total);

        gpuErrchk(cudaDeviceSynchronize());
    }

    void RigidParticles::calculate_overlapping_rigidbody(
        NodesContainer &nodes_ref,
        ParticlesContainer &particles_ref)
    {
        KERNEL_GET_OVERLAPPING_RIGID_BODY_GRID<<<nodes_ref.launch_config.tpb,
                                              nodes_ref.launch_config.bpg>>>(
            thrust::raw_pointer_cast(is_overlapping_gpu.data()),
            thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
            thrust::raw_pointer_cast(positions_gpu.data()),
            thrust::raw_pointer_cast(spatial.bins_gpu.data()),
            particles_ref.spatial.num_cells, particles_ref.spatial.grid_start,
            particles_ref.spatial.inv_cell_size,
            particles_ref.spatial.num_cells_total, num_particles);

        gpuErrchk(cudaDeviceSynchronize());
    }

    void RigidParticles::update_grid_moments(NodesContainer &nodes_ref,
                                             ParticlesContainer &particles_ref)
    {
        KERNEL_VELOCITY_CORRECTOR<<<nodes_ref.launch_config.tpb,
                                              nodes_ref.launch_config.bpg>>>(
            thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
            thrust::raw_pointer_cast(nodes_ref.moments_gpu.data()),
            thrust::raw_pointer_cast(closest_rigid_particle_gpu.data()),
            thrust::raw_pointer_cast(velocities_gpu.data()),
            thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
            thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()),
            thrust::raw_pointer_cast(normals_gpu.data()),
            thrust::raw_pointer_cast(is_overlapping_gpu.data()), rotation_matrix, COM,
            translational_velocity, particles_ref.spatial.grid_start,
            particles_ref.spatial.inv_cell_size,
            particles_ref.spatial.num_cells_total);
        gpuErrchk(cudaDeviceSynchronize());
    }

    void RigidParticles::update_rigid_body(NodesContainer &nodes_ref,
                                           ParticlesContainer &particles_ref)

    {
        KERNEL_UPDATE_POS_RIGID<<<nodes_ref.launch_config.tpb,
                                              nodes_ref.launch_config.bpg>>>(
            thrust::raw_pointer_cast(positions_gpu.data()),
            thrust::raw_pointer_cast(velocities_gpu.data()), rotation_matrix, COM,
            translational_velocity, num_particles);
        gpuErrchk(cudaDeviceSynchronize());
    }

    void RigidParticles::calculate_velocities()
    {
        const Vectorr COM_nt = locations_cpu[global_step_cpu];
        const Vectorr ROT_nt = rotations_cpu[global_step_cpu];

        translational_velocity = (COM_nt - COM) / dt_cpu;

        const Vectorr rotational_velocity = (ROT_nt - ROT);

#if DIM == 3
        const AngleAxisr rollAngle(rotational_velocity[0],
                                   Vectorr::UnitX()); // pitch

        const AngleAxisr yawAngle(rotational_velocity[1], Vectorr::UnitY()); // yaw

        const AngleAxisr pitchAngle(rotational_velocity[2],
                                    Vectorr::UnitZ()); // roll

        Quaternionr q = rollAngle * yawAngle * pitchAngle;

        // rotation_matrix = q.matrix();
        rotation_matrix = Matrixr::Zero();
#elif DIM == 2

        // const AngleAxisr rollAngle(rotational_velocity[0],Vectorr::UnitX()); // pitch

        // const AngleAxisr yawAngle(rotational_velocity[1], Vectorr::UnitY()); // yaw

        // Quaternionr q = rollAngle * yawAngle;

        // rotation_matrix = q.matrix();

#else
        rotation_matrix = Matrixr::Zero();
#endif

//           COM = COM_nt;
//           ROT = ROT_nt;
    }

    void RigidParticles::find_nearest_rigid_body(
        NodesContainer &nodes_ref,
        ParticlesContainer &particles_ref)
    {
        KERNEL_FIND_NEAREST_RIGIDPARTICLE<<<nodes_ref.launch_config.tpb,
                                              nodes_ref.launch_config.bpg>>>(
            thrust::raw_pointer_cast(closest_rigid_particle_gpu.data()),
            thrust::raw_pointer_cast(positions_gpu.data()),
            thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
            thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()),
            thrust::raw_pointer_cast(spatial.cell_start_gpu.data()),
            thrust::raw_pointer_cast(spatial.cell_end_gpu.data()),
            thrust::raw_pointer_cast(spatial.sorted_index_gpu.data()),
            thrust::raw_pointer_cast(is_overlapping_gpu.data()), spatial.num_cells,
            spatial.grid_start, spatial.inv_cell_size, spatial.num_cells_total);

        gpuErrchk(cudaDeviceSynchronize());
    };

    void RigidParticles::apply_on_nodes_moments(NodesContainer &nodes_ref,
                                                ParticlesContainer &particles_ref){
          // TODO move this somewhere else?
          if (global_step_cpu == 0) {
            initialize(nodes_ref, particles_ref);
          }

          partition();

          if (num_frames > 0) {
            calculate_velocities();  // host bound
          }

          calculate_overlapping_rigidbody(nodes_ref, particles_ref);

          calculate_non_rigid_grid_normals(nodes_ref, particles_ref);

          find_nearest_rigid_body(nodes_ref, particles_ref);

          if (num_frames > 0) {
            update_rigid_body(nodes_ref, particles_ref);
          }

          // std::cout << rotation_matrix << "\n -----------------------------\n";
          update_grid_moments(nodes_ref, particles_ref);
    };

    void RigidParticles::output_vtk()
    {
          vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

          set_vtk_points(positions_gpu, polydata);

          cpu_array<Vectorr> velocities_cpu = velocities_gpu;
          set_vtk_pointdata<Vectorr>(velocities_cpu, polydata, "velocities");

        //   // cpu_array<Vector3r> velocities_cpu = velocities_gpu;
        //   // set_vtk_pointdata(velocities_cpu, polydata, "Velocity");

        //   vtkSmartPointer<vtkTransform> transform =
        //   vtkSmartPointer<vtkTransform>::New();

        //   // transform->Translate(0.575, 0.4  , 0.);
        //   // transform->RotateZ(90.0);
        //   // transform->Translate(-0.575, -0.4  , -0.);

        //   // vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter =
        //   vtkSmartPointer<vtkTransformPolyDataFilter>::New();
        //   // transformFilter->SetInputData(polydata);
        //   // transformFilter->SetTransform(transform);
        //   // transformFilter->Update();

        //    vtkSmartPointer<vtkPolyData> polydata_transformed =
        //    vtkSmartPointer<vtkPolyData>::New();

        //   //  polydata_transformed = transformFilter->GetOutput();
        //   // transformFilter->

        //   write_vtk_polydata(polydata, "rigidbody");

        for (auto format : output_formats)
        {
            write_vtk_polydata(polydata, "rigidbody", format);
        }
    }

} // namespace pyroclastmpm