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

#pragma once

// #define CUDA_ENABLED

#ifdef CUDA_ENABLED
#include <thrust/device_vector.h>
#endif

#include <Eigen/Dense>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

namespace pyroclastmpm {

#ifndef USE_DOUBLES
using Real = float;
#else
using Real = double;
#endif

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
#define NODE_MEM_INDEX(BIN, NUM_BINS)                                          \
  (BIN[0] + BIN[1] * NUM_BINS[0] + BIN[2] * NUM_BINS[0] * NUM_BINS[1])
#define WINDOW_BIN(BIN, WINDOW, i)                                             \
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

#ifdef CUDA_ENABLED
template <typename T> using gpu_array = thrust::device_vector<T>;
#else
template <typename T> using gpu_array = thrust::host_vector<T>;
#endif

template <typename T> using cpu_array = thrust::host_vector<T>;

enum OutputType { VTK, OBJ, CSV, GTFL };

enum SFType {
  LinearShapeFunction = 0,
  QuadraticShapeFunction = 1,
  CubicShapeFunction = 2 // TODO Fix
};

// TODO is this needed?
enum BCType { NodeBoundaryCondition, ParticleBoundaryCondition };

constexpr size_t BLOCKSIZE = 64;
constexpr size_t WARPSIZE = 32;

#ifdef CUDA_ENABLED
// trunk-ignore-all(codespell/misspelled)
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " "
              << line << std::endl;

    if (abort)
      exit(code);
  }
}
struct GPULaunchConfig {
  /**
   * @brief The number of threads per block (Block size)
   *
   */
  dim3 tpb;

  /**
   * @brief The number of blocks per grid (Grid size)
   *
   */
  dim3 bpg;
};
#endif

} // namespace pyroclastmpm