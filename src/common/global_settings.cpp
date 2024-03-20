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
 * @file global_settings.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Contains global variables used throughout the code.
 * Normally, it is not good practice to use global variables, but in this case
 * it is necessary for performance reasons. CUDA __constant__ memory is stored
 * in the constant cache which is much faster than global memory.
 *
 * Important that global variables are only modified by the host, and set
 * in this file.
 *
 *
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "pyroclastmpm/common/global_settings.h"

namespace pyroclastmpm
{

#ifdef CUDA_ENABLED

  /// @brief Shape function type (DEVICE)
  __constant__ SFType shape_function_gpu = LinearShapeFunction;

  /// @brief number of surrounding nodes for grid/particle interpolation (DEVICE)
  __constant__ int num_surround_nodes_gpu;

  /// @brief Time step (DEVICE)
  __constant__ Real dt_gpu;

  /// @brief Global step counter (DEVICE)
  __constant__ int global_step_gpu = 0;

  /// @brief Initial number of particles per cell (DEVICE)
  __constant__ int particles_per_cell_gpu;

  /// @brief Connectivity window for grid to particle interpolation (DEVICE)
  __constant__ int g2p_window_gpu[64][3];

  /// @brief Connectivity window for particle to grid interpolation (DEVICE)
  __constant__ int p2g_window_gpu[64][3];
#endif

  /// @brief Global output directory string
  char output_directory_cpu[256];

  /// @brief Connectivity window for grid to particle interpolation
  int g2p_window_cpu[64][3];

  /// @brief Connectivity window for particle to grid interpolation
  int p2g_window_cpu[64][3];

  /// @brief Shape function type (HOST)
  SFType shape_function_cpu = LinearShapeFunction;

  /// @brief Number of surrounding nodes for grid/particle interpolation (HOST)
  int num_surround_nodes_cpu = 1;

  /// @brief Time step (HOST)
  Real dt_cpu = 0.0;

  /// @brief Inverse cell size (HOST)
  Real inv_cell_size_cpu = 0.0;

  /// @brief Global step counter (HOST)
  int global_step_cpu = 0;

  /// @brief Initial particles per cell (HOST)
  int particles_per_cell_cpu;

// TODO: check if these are consistent with old code... found some possible
// mistakes
/// Connectivity windows for grid to particle and particle to grid interpolation
/// These are run-time constants, and may be different for different shape or
/// dimension
#if DIM == 1
  /// @brief Connectivity window for linear grid to particle interpolation
  /// (Grid-To-Particle)
  const int linear_g2p_window[64][3] = {{0, 0, 0}, {1, 0, 0}};
  /// @brief Connectivity window for linear particle to grid interpolation
  /// (Particle-To-Grid)
  const int linear_p2g_window[64][3] = {{0, 0, 0}, {-1, 0, 0}};
  /// @brief Connectivity window for quadratic grid to particle interpolation
  /// (Grid-To-Particle)
  const int quadratic_g2p_window[64][3] = {
      {-1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
  /// @brief Connectivity window for quadratic particle to grid interpolation
  /// (Particle-To-Grid)
  const int quadratic_p2g_window[64][3] = {
      {1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {-2, 0, 0}};
#elif DIM == 2
  /// @brief Connectivity window for linear grid to particle interpolation
  /// (Grid-To-Particle)
  const int linear_g2p_window[64][3] = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  /// @brief Connectivity window for linear particle to grid interpolation
  /// (Particle-To-Grid)
  const int linear_p2g_window[64][3] = {
      {0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {-1, -1, 0}};
  /// @brief Connectivity window for quadratic grid to particle interpolation
  /// (Grid-To-Particle)
  const int quadratic_g2p_window[64][3] = {
      {-1, -1, 0}, {0, -1, 0}, {1, -1, 0}, {2, -1, 0}, {-1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {-1, 1, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {-1, 2, 0}, {0, 2, 0}, {1, 2, 0}, {2, 2, 0}};
  /// @brief Connectivity window for quadratic particle to grid interpolation
  /// (Particle-To-Grid)
  const int quadratic_p2g_window[64][3] = {
      {1, 1, 0}, {0, 1, 0}, {-1, 1, 0}, {-2, 1, 0}, {1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {-2, 0, 0}, {1, -1, 0}, {0, -1, 0}, {-1, -1, 0}, {-2, -1, 0}, {1, -2, 0}, {0, -2, 0}, {-1, -2, 0}, {-2, -2, 0}};
#else // DIM == 3
  /// @brief Connectivity window for linear grid to particle interpolation
  /// (Grid-To-Particle)
  const int linear_g2p_window[64][3] = {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 1, 1}};
  /// @brief Connectivity window for linear particle to grid interpolation
  /// (Particle-To-Grid)
  const int linear_p2g_window[64][3] = {{0, 0, 0}, {0, 0, -1}, {-1, 0, 0}, {-1, 0, -1}, {0, -1, 0}, {0, -1, -1}, {-1, -1, 0}, {-1, -1, -1}};
  /// @brief Connectivity window for quadratic grid to particle interpolation
  /// (Grid-To-Particle)
  const int quadratic_g2p_window[64][3] = {
      {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, -1, 2}, {0, -1, -1}, {0, -1, 0}, {0, -1, 1}, {0, -1, 2}, {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, -1, 2}, {2, -1, -1}, {2, -1, 0}, {2, -1, 1}, {2, -1, 2}, {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1}, {-1, 0, 2}, {0, 0, -1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {2, 0, -1}, {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1}, {-1, 1, 2}, {0, 1, -1}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {1, 1, -1}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {2, 1, -1}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {-1, 2, -1}, {-1, 2, 0}, {-1, 2, 1}, {-1, 2, 2}, {0, 2, -1}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2}, {1, 2, -1}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2}, {2, 2, -1}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2}};
  /// @brief Connectivity window for quadratic particle to grid interpolation
  /// (Particle-To-Grid)
  const int quadratic_p2g_window[64][3] = {
      {1, 1, 1}, {1, 1, 0}, {1, 1, -1}, {1, 1, -2}, {0, 1, 1}, {0, 1, 0}, {0, 1, -1}, {0, 1, -2}, {-1, 1, 1}, {-1, 1, 0}, {-1, 1, -1}, {-1, 1, -2}, {-2, 1, 1}, {-2, 1, 0}, {-2, 1, -1}, {-2, 1, -2}, {1, 0, 1}, {1, 0, 0}, {1, 0, -1}, {1, 0, -2}, {0, 0, 1}, {0, 0, 0}, {0, 0, -1}, {0, 0, -2}, {-1, 0, 1}, {-1, 0, 0}, {-1, 0, -1}, {-1, 0, -2}, {-2, 0, 1}, {-2, 0, 0}, {-2, 0, -1}, {-2, 0, -2}, {1, -1, 1}, {1, -1, 0}, {1, -1, -1}, {1, -1, -2}, {0, -1, 1}, {0, -1, 0}, {0, -1, -1}, {0, -1, -2}, {-1, -1, 1}, {-1, -1, 0}, {-1, -1, -1}, {-1, -1, -2}, {-2, -1, 1}, {-2, -1, 0}, {-2, -1, -1}, {-2, -1, -2}, {1, -2, 1}, {1, -2, 0}, {1, -2, -1}, {1, -2, -2}, {0, -2, 1}, {0, -2, 0}, {0, -2, -1}, {0, -2, -2}, {-1, -2, 1}, {-1, -2, 0}, {-1, -2, -1}, {-1, -2, -2}, {-2, -2, 1}, {-2, -2, 0}, {-2, -2, -1}, {-2, -2, -2}};
#endif

  /**
   * @brief Set the globals object
   * @param _dt time step
   * @param particles_per_cell number of particles per cell
   * @param _shapefunction shape function type ("linear" or "cubic"")
   * @param _output_dir output directory string
   */
  void set_globals(const Real _dt, const int particles_per_cell,
                   const std::string_view &_shapefunction,
                   const std::string &_output_dir)
  {
    bool isCudaEnabled = false;
#ifdef CUDA_ENABLED
    isCudaEnabled = true;
#endif
    spdlog::info("[Global] Use GPU: {}", isCudaEnabled);
    spdlog::info("[Global] Dimension compliled {}", DIM);

    set_global_dt(_dt);
    set_global_shapefunction(_shapefunction);
    set_global_output_directory(_output_dir);
    set_global_particles_per_cell(particles_per_cell);
  }

  /// @brief Set the global initial particles per cell.
  /// @param _particles_per_cell initial particles per cell
  void set_global_particles_per_cell(const int _particles_per_cell)
  {
    particles_per_cell_cpu = _particles_per_cell;
#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(particles_per_cell_gpu, &(_particles_per_cell),
                       sizeof(int), 0);
#endif
    spdlog::info("[Global] particles per cell: {}", _particles_per_cell);
  }

  /// @brief Set the global output directory object
  /// @param _output_dir output directory string
  void set_global_output_directory(const std::string_view &_output_dir)
  {
    std::copy(_output_dir.begin(), _output_dir.end(), output_directory_cpu);
    spdlog::info("[Global] output directory: {}", _output_dir);
  };

  /// @brief Set the global timestep
  /// @param _dt timestep
  void set_global_dt(const Real _dt)
  {
    dt_cpu = _dt;
#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(dt_gpu, &(_dt), sizeof(Real), 0);
#endif
    spdlog::info("[Global] time step: {}", _dt);
  };

  /// @brief Set the global step object
  /// @param _step global step counter
  void set_global_step(const int _step)
  {
    global_step_cpu = _step;
#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(global_step_gpu, &(_step), sizeof(int), 0);
#endif
  };

  /// @brief Increment the global step counter
  void increment_global()
  {
    global_step_cpu += 1;
#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(global_step_gpu, &(global_step_cpu), sizeof(int), 0);
#endif
  }

  /// @brief Set the global shapefunction enum type
  /// @details sets a window_size and num_surround_nodes as well
  /// which is important for traversing the connectivity windows
  /// @param _shapefunction ("linear" or "cubic")
  void set_global_shapefunction(const std::string_view &_shapefunction)
  {

    int window_size_cpu;
    if (_shapefunction == "linear")
    {
      shape_function_cpu = LinearShapeFunction;
      window_size_cpu = 2;
      num_surround_nodes_cpu = (int)pow((double)window_size_cpu, (double)DIM);
#ifdef CUDA_ENABLED
      cudaMemcpyToSymbol(g2p_window_gpu, linear_g2p_window, 64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(p2g_window_gpu, linear_p2g_window, 64 * 3 * sizeof(int));
#else
      std::copy(&linear_g2p_window[0][0], &linear_g2p_window[0][0] + 64 * 3,
                &g2p_window_cpu[0][0]);
      std::copy(&linear_p2g_window[0][0], &linear_p2g_window[0][0] + 64 * 3,
                &p2g_window_cpu[0][0]);
#endif
    }
    else if (_shapefunction == "cubic")
    {

      shape_function_cpu = CubicShapeFunction;
      window_size_cpu = 4;
      num_surround_nodes_cpu = (int)pow((double)window_size_cpu, (double)DIM);
#ifdef CUDA_ENABLED
      cudaMemcpyToSymbol(g2p_window_gpu, quadratic_g2p_window,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(p2g_window_gpu, quadratic_p2g_window,
                         64 * 3 * sizeof(int));
#else
      std::copy(&quadratic_g2p_window[0][0], &quadratic_g2p_window[0][0] + 64 * 3,
                &g2p_window_cpu[0][0]);
      std::copy(&quadratic_p2g_window[0][0], &quadratic_p2g_window[0][0] + 64 * 3,
                &p2g_window_cpu[0][0]);
#endif
    }
    else
    {
      std::cout << "Shape function not recognized, exiting." << std::endl;
      exit(1);
    }

#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(shape_function_gpu, &(shape_function_cpu), sizeof(int), 0);
    cudaMemcpyToSymbol(num_surround_nodes_gpu, &(num_surround_nodes_cpu),
                       sizeof(int), 0);
#endif
    spdlog::info("[Global] shape function: {}", _shapefunction);
  };

}; // namespace pyroclastmpm