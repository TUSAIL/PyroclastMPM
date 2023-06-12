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

#include "pyroclastmpm/solver/solver.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern __constant__ SFType shape_function_gpu;
extern __constant__ int num_surround_nodes_gpu;
extern __constant__ int forward_window_gpu[64][3];
#else
extern SFType shape_function_cpu;
extern int num_surround_nodes_cpu;
extern int forward_window_cpu[64][3];
#endif

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
 * (4) reorder the particles (TODO: broken)
 *
 * @param _particles particles container
 * @param _nodes nodes container
 * @param _boundaryconditions a list of boundary conditions to be applied
 * @param _materials a list of materials to be applied
 */
Solver::Solver(ParticlesContainer _particles, NodesContainer _nodes,
               cpu_array<MaterialType> _materials,
               cpu_array<BoundaryConditionType> _boundaryconditions)
    : particles(_particles), nodes(_nodes), materials(_materials),
      boundaryconditions(_boundaryconditions) {
  particles.set_spatialpartition(nodes.node_start, nodes.node_end,
                                 nodes.node_spacing);

  particles.calculate_initial_volumes();

  for (int mat_id = 0; mat_id < materials.size(); mat_id++) {
    std::visit(
        [&](auto &arg) {
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
void Solver::stress_update() {
  // todo make it so material can have different stress measure

  for (int mat_id = 0; mat_id < materials.size(); mat_id++) {
    std::visit([&](auto &arg) { arg.stress_update(particles, mat_id); },
               materials[mat_id]);
  }
}

/**
 * @brief Solve a n number of MPM iterations
 * @param n_steps
 */
void Solver::solve_nsteps(int n_steps) {
  for (int step = 0; step < n_steps; step++) {
    solve();
    ++global_step_cpu;
  }
  output();
}

/**
 * @brief Output the particles, nodes and boundary conditions
 *
 */
void Solver::output() {
  particles.output_vtk();
  nodes.output_vtk();
  // particles_ptr->reorder();
  for (auto &bc : boundaryconditions) {
    // bc.output_vtk();

    std::visit([&](auto &arg) { arg.output_vtk(); }, bc);
  }
}

/**
 * @brief Destroy the Solver:: Solver object
 *
 */
Solver::~Solver() { global_step_cpu = 0; }

} // namespace pyroclastmpm