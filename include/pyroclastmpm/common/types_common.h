// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
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
 * @file types_common.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Common types and definitions used in the MPM code.
 * @details Contains differnt types, structs and MACROs used in the MPM code.
 *
 * USE_DOUBLE is when the code is compiled with double precision.
 *
 * CUDA_ENABLED is when the code is compiled with CUDA enabled.
 *
 * DIM is the dimension of the problem (1, 2 or 3).
 *
 * TODO: fix doxygen documentation output
 *
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

/**
 * ... Thrust is device_arrays are on host if CUDA is not enabled ...
 */
#ifdef CUDA_ENABLED
#include <thrust/device_vector.h>
#endif

#include <Eigen/Dense>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include "spdlog/spdlog.h"
#include <spdlog/fmt/ostr.h>

namespace pyroclastmpm
{

#ifndef USE_DOUBLES
  using Real = float;
#else
  using Real = double;
#endif

  /**
   * ... `Vectorr`, `Matrixr` size is determined by the compiler flag `DIM`. ...
   * ... The suffix `r` denotes its a Real (float or double), while `i` denotes
   * integer ...
   */

  typedef unsigned char uint8_t;

  using Vector3i = Eigen::Matrix<int, 3, 1>;
  using Vector3r = Eigen::Matrix<Real, 3, 1>;
  using Matrix3r = Eigen::Matrix<Real, 3, 3>;
  using Vector3b = Eigen::Matrix<bool, 3, 1>;

  using Vector2i = Eigen::Matrix<int, 2, 1>;
  using Vector2r = Eigen::Matrix<Real, 2, 1>;
  using Matrix2r = Eigen::Matrix<Real, 2, 2>;
  using Vector2b = Eigen::Matrix<bool, 2, 1>;

  using Vector1i = Eigen::Matrix<int, 1, 1>;
  using Vector1r = Eigen::Matrix<Real, 1, 1>;
  using Matrix1r = Eigen::Matrix<Real, 1, 1>;
  using Vector1b = Eigen::Matrix<bool, 1, 1>;

#if DIM == 3

/// Macro to get cartesian index from a 3D bin index
#define NODE_MEM_INDEX(BIN, NUM_BINS) \
  (BIN[0] + BIN[1] * NUM_BINS[0] + BIN[2] * NUM_BINS[0] * NUM_BINS[1])

/// Macro to calculate the bin index from a window index (used in p2g, g2p)
#define WINDOW_BIN(BIN, WINDOW, i) \
  (BIN + Vectori({WINDOW[i][0], WINDOW[i][1], WINDOW[i][2]}))

  using Vectorr = Vector3r;
  using Matrixr = Matrix3r;
  using Vectori = Vector3i;
#elif DIM == 2

#define NODE_MEM_INDEX(BIN, NUM_BINS) (BIN[0] + BIN[1] * NUM_BINS[0])
#define WINDOW_BIN(BIN, WINDOW, i) (BIN + Vectori({WINDOW[i][0], WINDOW[i][1]}))
  using Vectorr = Vector2r;
  using Matrixr = Matrix2r;
  using Vectori = Vector2i;
#else // DIM == 1
#define NODE_MEM_INDEX(BIN, NUM_BINS) (BIN[0])
#define WINDOW_BIN(BIN, WINDOW, i) (BIN + Vectori(WINDOW[i][0]))
  using Vectorr = Vector1r;
  using Matrixr = Matrix1r;
  using Vectori = Vector1i;
#endif

  using Quaternionr = Eigen::Quaternion<Real>;
  using AngleAxisr = Eigen::AngleAxis<Real>;

  constexpr double PI = 3.14159265358979323846;

  /**
   *
   * ... Thrust stl containers are used for memory management ...
   * ... `gpu_array` is a device array if CUDA is enabled, otherwise its a host
   * array ...
   * ... `cpu_array` is always a host array ...
   */

#ifdef CUDA_ENABLED
  template <typename T>
  using gpu_array = thrust::device_vector<T>;
#else
  template <typename T>
  using gpu_array = thrust::host_vector<T>;
#endif

  template <typename T>
  using cpu_array = thrust::host_vector<T>;

  /**
   * @brief Shape function type
   * @details this also determines the connectivity of the particles.
   */
  enum SFType
  {
    LinearShapeFunction = 0,
    QuadraticShapeFunction = 1,
    CubicShapeFunction = 2
  };

  /// @brief launch block configuration of CUDA kernels
  constexpr size_t BLOCKSIZE = 64;

  /// @brief launch wapr size configuration of CUDA kernels
  constexpr size_t WARPSIZE = 32;

#ifdef CUDA_ENABLED
/**
 * @brief MACRO used to check if an error occured on the GPU
 *
 */
// trunk-ignore-all(codespell/misspelled)
#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
  inline void gpuAssert(cudaError_t code, const char *file, int line,
                        bool abort = true)
  {
    if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
              line);
      std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " "
                << line << std::endl;

      if (abort)
        exit(code);
    }
  }

  /**
   * @brief Launch configuration for CUDA kernels
   * \verbatim embed:rst:leading-asterisk
   *     Example usage
   *
   *     .. code-block:: cpp
   *
   *         #include "pyroclastmpm/common/types_common.h"
   *
   *         GPULaunchConfig launch_config(num_elements);
   *
   *         SomeKernel<<<nodes.launch_config.tpb,
   * nodes.launch_config.bpg>>>(...); gpuErrchk(cudaDeviceSynchronize());
   *
   * \endverbatim
   */
  struct GPULaunchConfig
  {

    /// @brief default constructor
    GPULaunchConfig() = default;

    /// @brief Construct a new GPULaunchConfig object
    /// @param _tpb threads per block
    /// @param _bpg blocks per grid
    GPULaunchConfig(const dim3 _tpb, const dim3 _bpg) : tpb(_tpb), bpg(_bpg) {}

    /// @brief Construct a new GPULaunchConfig object
    /// @param num_elements number of elements to be processed
    GPULaunchConfig(const int num_elements)
    {
      tpb = dim3(int((num_elements) / BLOCKSIZE) + 1, 1, 1);
      bpg = dim3(BLOCKSIZE, 1, 1);
      gpuErrchk(cudaDeviceSynchronize());
    }
    /// number of threads per block (Block size)
    dim3 tpb;

    /// number of blocks per grid (Grid size)
    dim3 bpg;
  };
#endif

  /**
   * @brief Grid data structure.
   * This is used in nodes and spatial partitioning.
   * \verbatim embed:rst:leading-asterisk
   *     Example usage
   *
   *     .. code-block:: cpp
   *
   *         #include "pyroclastmpm/common/types_common.h"
   *
   *         Vectorr origin = Vectorr::Zero();
   *         Vectorr end = Vectorr::Ones();
   *         const Real cell_size = 0.1;
   *
   *         Grid grid(origin, end, cell_size);
   *
   * \endverbatim
   */
  __device__ __host__ struct Grid
  {

    /// @brief default constructor
    __device__ __host__ Grid() = default;

    // / @brief Construct a new Grid object
    // / @param _origin start coordinates of the partitioning grid
    // / @param _end end coordinates of the partitioning grid
    // / @param _cell_size cells size of the partitioning grid
    __device__ __host__ Grid(const Vectorr _origin, const Vectorr _end,
                             const Real _cell_size)
        : origin(_origin), end(_end), cell_size(_cell_size)
    {

      inv_cell_size = (Real)1.0 / cell_size;

      for (int axis = 0; axis < DIM; axis++)
      {
        num_cells[axis] = (int)((end[axis] - origin[axis]) / cell_size) + 1;
        num_cells_total *= num_cells[axis];
      }
    }

    /// @brief start domain of the partitioning grid
    Vectorr origin;

    /// @brief end domain of the partitioning grid
    Vectorr end;

    /// @brief cell size of the partitioning grid
    Real cell_size;

    /// @brief inverse cell size of the partitioning grid
    Real inv_cell_size;

    /// @brief number of cells
    Vectori num_cells = Vectori::Ones();

    /// @brief total number of cells within the grid
    int num_cells_total = 1;
  };

} // namespace pyroclastmpm