#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <Eigen/Dense>
#include <iostream>

namespace pyroclastmpm {

#define DIM 2

#define USE_DOUBLES

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
using Vectorr = Vector3r;
using Matrixr = Matrix3r;
using Vectori = Vector3i;
#elif DIM == 2
using Vectorr = Vector2r;
using Matrixr = Matrix2r;
using Vectori = Vector2i;
#else // DIM == 1
using Vectorr = Vector1r;
using Matrixr = Matrix1r;
using Vectori = Vector1i;
#endif

using Quaternionr = Eigen::Quaternion<Real>;
using AngleAxisr  = Eigen::AngleAxis<Real>;

template <typename T> using gpu_array = thrust::device_vector<T>;
template <typename T> using cpu_array = thrust::host_vector<T>;

enum OutputType { VTK, OBJ, CSV, HDF5 };

enum StressMeasure { PK1, PK2, Cauchy, Kirchhoff, None }; //TODO remove

enum SFType {
  LinearShapeFunction = 0,
  QuadraticShapeFunction = 1,
  CubicShapeFunction = 2
};

// TODO is this needed?
enum BCType { NodeBoundaryCondition, ParticleBoundaryCondition };

#define BLOCKSIZE 64
#define WARPSIZE 32

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
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

}  // namespace pyroclastmpm