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
 * @file helper.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Implementation of helper functions for host and device arrays
 * @details Thrust dependent helper functions for host and device arrays
 * significantly reduce the amount of code needed for memory management.
 *
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "pyroclastmpm/common/helper.h"
#include <thrust/gather.h>

namespace pyroclastmpm
{

  ///@brief  Helper function to allocate device memory
  ///@tparam T data type (float, double, int, Matrixr, etc.)
  ///@param input_size input_size size of the input array
  ///@param input input_size size of the input array
  ///@param output output output array (of device type
  ///@param default_value default_value default value to set the array to
  template <typename T>
  double set_default_device(const int input_size, const cpu_array<T> input,
                            gpu_array<T> &output, T default_value)
  {
    // If input is empty, do not copy and set output to default value instead

    if (input.empty())
    {
      cpu_array<T> output_host;

      output_host.resize(input_size);
      for (auto &out : output_host)
      {
        out = default_value;
      }
      output = output_host;
    }
    else
    {
      output = input;
    }

    // Calculate size in bytes
    size_t size_in_bytes = input_size * sizeof(T);
    // Convert to megabytes
    double size_in_mb = static_cast<double>(size_in_bytes) / (1024 * 1024);
    return size_in_mb;
  }

  /// @brief  Helper function to allocate device memory
  /// @details Uses the thrust gather interface to reorder a device.
  /// @tparam T data type (float, double, int, Matrixr, etc.)
  /// @param input_size input_size size of the input array
  /// @param input input_size size of the input array
  /// @param output output output array (of device type
  /// @param default_value default_value default value to set the array to
  template <typename T>
  void reorder_device_array(gpu_array<T> &output, gpu_array<int> sorted_index)
  {
    gpu_array<T> device_temp = output;
    thrust::gather(sorted_index.begin(), sorted_index.end(), device_temp.begin(),
                   output.begin());
  }

  /// @brief Helper function to print a device or host array
  /// @tparam T data type (float, double, int, Matrixr, etc.)
  /// @param input array to be printed
  template <typename T>
  void print_array(const cpu_array<T> input)
  {
    std::cout << std::endl;
    for (auto &out : input)
    {

      if constexpr (std::is_same_v<T, Matrixr> || std::is_same_v<T, Vectorr> ||
                    std::is_same_v<T, Matrix3r> || std::is_same_v<T, Vectori>)
      {
        std::string sep = "\n----------------------------------------\n";
        Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        std::cout << out.format(CleanFmt) << sep;
      }
      else
      {
        std::cout << out << ", ";
      }
    }
    std::cout << std::endl;
  }

  // Explicitly initiate the template functions for the following types
  template void print_array<bool>(const cpu_array<bool> input);

  template double set_default_device<bool>(const int input_size,
                                           const cpu_array<bool> input,
                                           gpu_array<bool> &output,
                                           bool default_value);

  template void reorder_device_array<bool>(gpu_array<bool> &output,
                                           gpu_array<int> sorted_index);

  template void print_array<uint8_t>(const cpu_array<uint8_t> input);

  template double set_default_device<uint8_t>(const int input_size,
                                              const cpu_array<uint8_t> input,
                                              gpu_array<uint8_t> &output,
                                              uint8_t default_value);

  template void reorder_device_array<uint8_t>(gpu_array<uint8_t> &output,
                                              gpu_array<int> sorted_index);

  template void print_array<unsigned int>(const cpu_array<unsigned int> input);

  template double set_default_device<unsigned int>(
      const int input_size, const cpu_array<unsigned int> input,
      gpu_array<unsigned int> &output, unsigned int default_value);

  template void
  reorder_device_array<unsigned int>(gpu_array<unsigned int> &output,
                                     gpu_array<int> sorted_index);

  template void print_array<int>(const cpu_array<int> input);

  template double set_default_device<int>(const int input_size,
                                          const cpu_array<int> input,
                                          gpu_array<int> &output,
                                          int default_value);

  template void reorder_device_array<int>(gpu_array<int> &output,
                                          gpu_array<int> sorted_index);

  template void print_array<Real>(const cpu_array<Real> input);

  template double set_default_device<Real>(const int input_size,
                                           const cpu_array<Real> input,
                                           gpu_array<Real> &output,
                                           Real default_value);

  template void reorder_device_array<Real>(gpu_array<Real> &output,
                                           gpu_array<int> sorted_index);

  template void print_array<Matrixr>(const cpu_array<Matrixr> input);

  template double set_default_device<Matrixr>(const int input_size,
                                              const cpu_array<Matrixr> input,
                                              gpu_array<Matrixr> &output,
                                              Matrixr default_value);

  template void reorder_device_array<Matrixr>(gpu_array<Matrixr> &output,
                                              gpu_array<int> sorted_index);

  template void print_array<Vectori>(const cpu_array<Vectori> input);

  template double set_default_device<Vectori>(const int input_size,
                                              const cpu_array<Vectori> input,
                                              gpu_array<Vectori> &output,
                                              Vectori default_value);

  template void reorder_device_array<Vectori>(gpu_array<Vectori> &output,
                                              gpu_array<int> sorted_index);

// Note Matri3r is a typedef of Matrixr is same as Vector
#if DIM != 1
  template double set_default_device<Vectorr>(const int input_size,
                                              const cpu_array<Vectorr> input,
                                              gpu_array<Vectorr> &output,
                                              Vectorr default_value);

  template void reorder_device_array<Vectorr>(gpu_array<Vectorr> &output,
                                              gpu_array<int> sorted_index);

  template void print_array<Vectorr>(const cpu_array<Vectorr> input);
#endif
#if DIM != 3
  template double set_default_device<Matrix3r>(const int input_size,
                                               const cpu_array<Matrix3r> input,
                                               gpu_array<Matrix3r> &output,
                                               Matrix3r default_value);

  template void reorder_device_array<Matrix3r>(gpu_array<Matrix3r> &output,
                                               gpu_array<int> sorted_index);

  template void print_array<Matrix3r>(const cpu_array<Matrix3r> input);
#endif
} // namespace pyroclastmpm