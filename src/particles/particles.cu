#include "pyroclastmpm/particles/particles.cuh"

namespace pyroclastmpm
{

    // number of nodes surrounding each element
    extern int num_surround_nodes_cpu;
    extern int particles_per_cell_cpu;

    /*!
     * @brief Constructs Particle container class
     * @param _positions particle positions
     * @param _velocities particle velocities
     * @param _colors particle types (optional)
     * @param _stresses particle stresses (optional)
     * @param _masses particle masses (optional)
     * @param _volumes particle volumes (optional)
     */
    ParticlesContainer::ParticlesContainer(
        const cpu_array<Vectorr> _positions,
        const cpu_array<Vectorr> _velocities,
        const cpu_array<uint8_t> _colors,
        const cpu_array<Matrix3r> _stresses,
        const cpu_array<Real> _masses,
        const cpu_array<Real> _volumes,
        const cpu_array<OutputType> _output_formats)
    {

        output_formats = _output_formats;
        num_particles = _positions.size();

        set_default_device<Matrix3r>(num_particles, _stresses, stresses_gpu, Matrix3r::Zero());
        set_default_device<Vectorr>(num_particles, _positions, positions_gpu, Vectorr::Zero());
        set_default_device<Vectorr>(num_particles, _velocities, velocities_gpu, Vectorr::Zero());
        set_default_device<uint8_t>(num_particles, _colors, colors_gpu, 0);
        set_default_device<Real>(num_particles, _masses, masses_gpu, -1.0);
        set_default_device<Real>(num_particles, _volumes, volumes_gpu, -1.0);

        set_default_device<Matrixr>(num_particles, {}, velocity_gradient_gpu, Matrixr::Zero());
        set_default_device<Matrixr>(num_particles, {}, F_gpu, Matrixr::Identity());
        set_default_device<Vectorr>(num_surround_nodes_cpu * num_particles, {}, dpsi_gpu, Vectorr::Zero());
        set_default_device<Real>(num_particles, _volumes, volumes_original_gpu, -1.0);
        set_default_device<Real>(num_surround_nodes_cpu * num_particles, {}, psi_gpu, 0.0);

        set_default_device<Vectorr>(num_particles, {}, forces_external_gpu, Vectorr::Zero());
        // Visuals
        // TODO we need to find a way to store these optionally . . hopefully this
        // becomes more clear later
        set_default_device<uint8_t>(num_particles, {}, phases_gpu, 0);
        set_default_device<Matrixr>(num_particles, {}, strain_increments_gpu,
                                    Matrixr::Zero());
        set_default_device<Real>(num_particles, {}, densities_gpu, 0.0);
        set_default_device<Real>(num_particles, {}, pressures_gpu, 0.0);

        /*Drucker prager*/
        set_default_device<Real>(num_particles, {}, logJp_gpu, 0.0);

        /* Non Local Granular fluidity */
        /*! * @brief plastic deformation matrix */
        set_default_device<Matrixr>(num_particles, {}, FP_gpu, Matrixr::Identity());
        set_default_device<Real>(num_particles, {}, mu_gpu, 0.0);
        set_default_device<Real>(num_particles, {}, g_gpu, 0.0);
        set_default_device<Real>(num_particles, {}, ddg_gpu, 0.0);

        spatial = SpatialPartition(); // create a temporary partitioning object, since we are getting domain size

        launch_config.tpb = dim3(int((num_particles) / BLOCKSIZE) + 1, 1, 1);
        launch_config.bpg = dim3(BLOCKSIZE, 1, 1);

        gpuErrchk(cudaDeviceSynchronize());

        reset(); // reset needed
    }

    void ParticlesContainer::reset(bool reset_psi)
    {

        if (reset_psi)
        {
            thrust::fill(thrust::device, psi_gpu.begin(), psi_gpu.end(), 0.);
            thrust::fill(thrust::device, dpsi_gpu.begin(), dpsi_gpu.end(),
                         Vectorr::Zero());
        }

        thrust::fill(thrust::device, velocity_gradient_gpu.begin(),
                     velocity_gradient_gpu.end(), Matrixr::Zero());

        thrust::fill(thrust::device, pressures_gpu.begin(), pressures_gpu.end(), 0.);

        thrust::fill(thrust::device, phases_gpu.begin(), phases_gpu.end(), -1);
        thrust::fill(thrust::device, forces_external_gpu.begin(), forces_external_gpu.end(), Vectorr::Zero());

    }

    void ParticlesContainer::reorder()
    {
        // TODO fix reordering
        printf("Reorder not working correctly \n");
        reorder_device_array<Vectorr>(positions_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Vectorr>(velocities_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Matrix3r>(stresses_gpu, spatial.sorted_index_gpu);

        reorder_device_array<Matrixr>(velocity_gradient_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Matrixr>(F_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Vectorr>(dpsi_gpu, spatial.sorted_index_gpu);
        reorder_device_array<uint8_t>(colors_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(volumes_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(volumes_original_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(masses_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(psi_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Vectorr>(forces_external_gpu, spatial.sorted_index_gpu);

        reorder_device_array<Real>(densities_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(pressures_gpu, spatial.sorted_index_gpu);
        reorder_device_array<uint8_t>(phases_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Matrixr>(strain_increments_gpu, spatial.sorted_index_gpu);

        reorder_device_array<Matrixr>(FP_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(mu_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(g_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(ddg_gpu, spatial.sorted_index_gpu);
        reorder_device_array<Real>(logJp_gpu, spatial.sorted_index_gpu);
    }

    void ParticlesContainer::output_vtk()
    {

        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

        cpu_array<Matrix3r> stresses_cpu = stresses_gpu;
        cpu_array<Matrixr> velocity_gradient_cpu = velocity_gradient_gpu;
        cpu_array<Matrixr> F_cpu = F_gpu;
        cpu_array<Matrixr> Fp_cpu = FP_gpu;
        cpu_array<Matrixr> strain_increments_cpu = strain_increments_gpu;
        cpu_array<Vectorr> velocities_cpu = velocities_gpu;
        cpu_array<Vectorr> positions_cpu = positions_gpu;
        cpu_array<Real> masses_cpu = masses_gpu;
        cpu_array<Real> volumes_cpu = volumes_gpu;
        cpu_array<Real> volumes_original_cpu = volumes_original_gpu;
        cpu_array<Real> densities_cpu = densities_gpu;
        cpu_array<Real> mu_cpu = mu_gpu;
        cpu_array<Real> g_cpu = g_gpu;
        cpu_array<Real> ddg_cpu = ddg_gpu;
        cpu_array<int> colors_cpu = colors_gpu;
        cpu_array<int> phases_cpu = phases_gpu;

        // get pressure
        cpu_array<Real> pressures_cpu;
        pressures_cpu.resize(num_particles);

        for (int pi = 0; pi < num_particles; pi++)
        {
            pressures_cpu[pi] = -(stresses_cpu[pi].block(0, 0, DIM, DIM).trace() / DIM);
        }

        set_vtk_points(positions_cpu, polydata);
        set_vtk_pointdata<Vectorr>(positions_cpu, polydata, "Positions");
        set_vtk_pointdata<Vectorr>(velocities_cpu, polydata, "Velocity");
        set_vtk_pointdata<Matrix3r>(stresses_cpu, polydata, "Stress");
        set_vtk_pointdata<Matrixr>(velocity_gradient_cpu, polydata, "VelocityGradient");
        set_vtk_pointdata<Matrixr>(F_cpu, polydata, "DeformationMatrix");
        set_vtk_pointdata<Real>(masses_cpu, polydata, "Mass");
        set_vtk_pointdata<Real>(volumes_cpu, polydata, "Volume");
        set_vtk_pointdata<Real>(volumes_original_cpu, polydata, "VolumeOriginal");
        set_vtk_pointdata<uint8_t>(colors_cpu, polydata, "Color");
        set_vtk_pointdata<Matrixr>(strain_increments_cpu, polydata, "Strain_Increments");
        set_vtk_pointdata<Real>(densities_cpu, polydata, "Density");
        set_vtk_pointdata<uint8_t>(phases_cpu, polydata, "Phase");
        set_vtk_pointdata<Real>(pressures_cpu, polydata, "Pressure");
        set_vtk_pointdata<Matrixr>(Fp_cpu, polydata, "PlasticDeformation");
        set_vtk_pointdata<Real>(mu_cpu, polydata, "FrictionCoef");
        set_vtk_pointdata<Real>(g_cpu, polydata, "G");
        set_vtk_pointdata<Real>(ddg_cpu, polydata, "DDG");

        // loop over output_formats
        for (auto format : output_formats)
        {
            write_vtk_polydata(polydata, "particles", format);
        }
    }

    void ParticlesContainer::set_spatialpartition(const Vectorr start,
                                                  const Vectorr end,
                                                  const Real spacing)
    {
        spatial = SpatialPartition(start, end, spacing, num_particles);
        partition();
    }

    void ParticlesContainer::partition()
    {
        spatial.calculate_hash(positions_gpu);

        spatial.sort_hashes();

        spatial.bin_particles();
    };

    void ParticlesContainer::calculate_initial_volumes()
    {
        if (isRestart)
        {
            return;
        }
        cpu_array<Real> volumes_cpu = volumes_gpu;
        for (int pi = 0; pi < num_particles; pi++)
        {

            if (volumes_cpu[pi] > 0.)
            {
                continue;
            }
#if DIM == 3
            volumes_cpu[pi] = spatial.cell_size * spatial.cell_size * spatial.cell_size;
#elif DIM == 2
            volumes_cpu[pi] = spatial.cell_size * spatial.cell_size;
#else
            volumes_cpu[pi] = spatial.cell_size;
#endif
            volumes_cpu[pi] /= particles_per_cell_cpu;
        }
        volumes_gpu = volumes_cpu;
        volumes_original_gpu = volumes_cpu;
    }

    void ParticlesContainer::calculate_initial_masses(int mat_id, Real density)
    {
        if (isRestart)
        {
            return;
        }
        cpu_array<int> colors_cpu = colors_gpu;
        cpu_array<Real> masses_cpu = masses_gpu;
        cpu_array<Real> volumes_cpu = volumes_gpu;

        for (int pi = 0; pi < num_particles; pi++)
        {
            if ((colors_cpu[pi] != mat_id) || (masses_cpu[pi] > 0.))
            {
                continue;
            }

            masses_cpu[pi] = density * volumes_cpu[pi];
        }

        masses_gpu = masses_cpu;
    }

} // namespace pyroclastmpm