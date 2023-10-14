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
 * @file solver.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Solver base class combines all the components of the MPM solver.
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#include "pyroclastmpm/solver/solver.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern __constant__ SFType shape_function_gpu;
extern __constant__ int num_surround_nodes_gpu;
extern __constant__ int forward_window_gpu[64][3];
#else
extern const SFType shape_function_cpu;
extern const int num_surround_nodes_cpu;
extern const int forward_window_cpu[64][3];
#endif

extern const int global_step_cpu;

/*!
 * @brief Construct a new Solver
 * home/retief/Code/TUSAIL/PyroclastMPM/ext/eigen/Eigen/Core:284,object
 * @details The following steps are performed:
 * (1) initialize the particles spatial partitioning
 * (2) calculate the initial volumes of the particles
 * (3) calculate the initial masses of the particles
 * (4) reorder the particles (TODO: broken)
 *
 * @param _particles ParticlesContainer
 * @param _nodes NodesContainer
 * @param _boundaryconditions A list of boundary conditions to be applied
 * @param _materials A list of materials to be applied
 */
Solver::Solver(const ParticlesContainer &_particles,
               const NodesContainer &_nodes,
               const cpu_array<MaterialType> &_materials,
               const cpu_array<BoundaryConditionType> &_boundaryconditions)
    : nodes(_nodes), particles(_particles), materials(_materials),
      boundaryconditions(_boundaryconditions) {
  particles.set_spatialpartition(nodes.grid);

  particles.calculate_initial_volumes();

  for (int mat_id = 0; mat_id < materials.size(); mat_id++) {
    std::visit(
        [this, mat_id](auto &arg) {
          particles.calculate_initial_masses(mat_id, arg.density);
        },
        materials[mat_id]);
  }

  particles.numColors = (int)materials.size();
  // TODO: reorder particles with particles.reorder(_)
  output();
}

/// @brief Do stress update for all the materials
/// @details loops through a list of variant materials and calls the
/// stress_update function for each material
void Solver::stress_update() {
  // todo make it so material can have different stress measure

  for (int mat_id = 0; mat_id < materials.size(); mat_id++) {
    std::visit(
        [this, mat_id](auto &arg) { arg.stress_update(particles, mat_id); },
        materials[mat_id]);
  }
}

/// @brief Solve the main loop for n_steps
/// @param n_steps
void Solver::solve_nsteps(int n_steps) {
  for (int step = 0; step < n_steps; step++) {
    solve();
    // Modifies global memory

    increment_global();
  }
  output();
}

/// @brief Solve the main loop for n_steps
/// @param total_steps number of steps to solve for
/// @param output_frequency output frequency
void Solver::run(const int total_steps, const int output_frequency) {

  // using namespace indicators;
  // ProgressBar bar{option::BarWidth{50},
  //                 option::Start{"⏳️["},
  //                 option::Fill{"."},
  //                 option::Lead{"■"},
  //                 option::Remainder{" "},
  //                 option::End{" ]"},
  //                 option::ShowElapsedTime{true},
  //                 option::ShowRemainingTime{true},
  //                 option::PrefixText{"Progress: "},
  //                 option::MaxProgress{total_steps + 1},
  //                 option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};
  printf("Running simulation...\n");

  output();

  solve();

  increment_global();

  for (int step = 1; step < total_steps + 1; step++) {
    solve();

    if (step % output_frequency == 0) {
      output();
      printf("Step %d/%d\n", step, total_steps);
      // // Show iteration as postfix text
      // bar.set_option(option::PostfixText{std::to_string(step) + "/" +
      //                                    std::to_string(total_steps)});

      // // update progress bar
      // bar.set_progress(step);
    }

    // Modifies global memory
    increment_global();
  }

  // bar.mark_as_completed();

  printf("Done.\n");

  // Show cursor
  // indicators::show_console_cursor(true);
};

/// @brief Output the results (ParticlesContainer,NodesContainer, etc. )
void Solver::output() {
  particles.output_vtk();
  nodes.output_vtk();

  for (auto &bc : boundaryconditions) {
    // FIXME: does not print out boundary condition child arrays
    // (is_overlapping) std::visit([this](auto &arg) { arg.output_vtk(nodes,
    // particles); }, bc);
  }
}

/// @brief Destroy the Solver:: Solver object
Solver::~Solver() { set_global_step(0); }

} // namespace pyroclastmpm