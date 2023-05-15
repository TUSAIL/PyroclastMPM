#include "pyroclastmpm/solver/solver.cuh"

namespace pyroclastmpm
{

  /**
   * @brief global step counter for the cpu
   *
   */
  extern int global_step_cpu;

  /*!
   * @brief Construct a new Solver:: Solver object also
   * (1) initialize the particles spatial partitioning
   * (2) calculate the initial volumes of the particles
   * (3) calculate the initial masses of the particles
   * (4) reorder the particles (TOOD: broken)
   *
   * @param _particles particles container
   * @param _nodes nodes container
   * @param _boundaryconditions a list of boundary conditions to be applied
   * @param _materials a list of materials to be applied
   */
  Solver::Solver(ParticlesContainer _particles,
                 NodesContainer _nodes,
                 cpu_array<MaterialType> _materials,
                 cpu_array<BoundaryConditionType> _boundaryconditions)
  {

    particles = _particles;

    nodes = _nodes;

    boundaryconditions = _boundaryconditions;

    materials = _materials;

    particles.set_spatialpartition(nodes.node_start, nodes.node_end,
                                   nodes.node_spacing);

    particles.calculate_initial_volumes();

    for (int mat_id = 0; mat_id < materials.size(); mat_id++)
    {
      std::visit(
          [&](auto &arg)
          {
            particles.calculate_initial_masses(mat_id, arg.density);
          },
          materials[mat_id]);
    }

    particles.numColors = materials.size();
    // particles.reorder(); // broken ? warning fails tests for instance
    output();
  }

  /**
   * @brief Solve a stress update step for the particles and their materials
   *
   */
  void Solver::stress_update()
  {
    // todo make it so material can have different stress meassure

    for (int mat_id = 0; mat_id < materials.size(); mat_id++)
    {

      std::visit([&](auto &arg)
                 {
                   // printf(" particles stress_measure (before 1st convert) %d \n",particles.stress_measure);
                   //  particles.convert_stress_measure(arg.stress_measure);
                   //  printf(" particles stress_measure (before stress update) %d \n",particles.stress_measure);
                   arg.stress_update(particles, mat_id);
                   //  particles.convert_stress_measure(stress_measure); // TODO attached color/id to stress conversion
                   //  printf(" particles stress_measure  (after second convert) %d\n",particles.stress_measure);
                   //  printf("material stress measure %d solver stress measure %d  \n", arg.stress_measure,stress_measure);
                 },
                 materials[mat_id]);
    }
  }

  /**
   * @brief Solve a n number of MPM iterations
   * @param n_steps
   */
  void Solver::solve_nsteps(int n_steps)
  {
    for (int step = 0; step < n_steps; step++)
    {
      solve();
      ++global_step_cpu;
    }
    output();
  }

  /**
   * @brief Output the particles, nodes and boundary conditions
   *
   */
  void Solver::output()
  {
    particles.output_vtk();
    nodes.output_vtk();
    // particles_ptr->reorder();
    for (auto &bc : boundaryconditions)
    {
      // bc.output_vtk();

      std::visit([&](auto &arg)
                 {
                   // printf(" particles stress_measure (before 1st convert) %d \n",particles.stress_measure);
                   //  particles.convert_stress_measure(arg.stress_measure);
                   //  printf(" particles stress_measure (before stress update) %d \n",particles.stress_measure);
                   arg.output_vtk();
                   //  particles.convert_stress_measure(stress_measure); // TODO attached color/id to stress conversion
                   //  printf(" particles stress_measure  (after second convert) %d\n",particles.stress_measure);
                   //  printf("material stress measure %d solver stress measure %d  \n", arg.stress_measure,stress_measure);
                 },
                 bc);
    }
  }

  /**
   * @brief Calculate the shapefunction values (and their gradients) for the particles and surrounding nodes
   *
   */
  void Solver::calculate_shape_function()
  {
    KERNEL_CALC_SHP<<<particles.launch_config.tpb, particles.launch_config.bpg>>>(
        thrust::raw_pointer_cast(particles.dpsi_gpu.data()),
        thrust::raw_pointer_cast(particles.psi_gpu.data()),
        thrust::raw_pointer_cast(particles.positions_gpu.data()),
        thrust::raw_pointer_cast(particles.spatial.bins_gpu.data()),
        thrust::raw_pointer_cast(nodes.node_types_gpu.data()), nodes.num_nodes,
        nodes.node_start, nodes.inv_node_spacing, particles.num_particles,
        nodes.num_nodes_total);
    gpuErrchk(cudaDeviceSynchronize());
  };

  /**
   * @brief Destroy the Solver:: Solver object
   *
   */
  Solver::~Solver()
  {

    global_step_cpu = 0;
  }

} // namespace pyroclastmpm