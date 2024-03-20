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

/**
 * @file helper.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief This header file contains helper functions for the MPM
 * code.
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm {

/**
 * @brief  Helper function to allocate device memory
 * @details Allocates device memory and assigns it a default host array or
 * constant value. \verbatim embed:rst:leading-asterisk Example usage
 *
 *     .. code-block:: cpp
 *
 *         #include "pyroclastmpm/common/helper.h"
 *
 *         // Resize array to num and assigns default value -1 to all ids
 *         set_default_device<int>(num, {}, ids, -1);
 *
 *         //  Copies a host array to a device array
 *         set_default_device<int>(num, ids_, ids, 0);
 *
 * \endverbatim
 * @tparam T data type (float, double, int, Matrixr, etc.)
 * @param input_size input_size size of the input array
 * @param input input_size size of the input array
 * @param output output output array (of device type
 * @param default_value default_value default value to set the array to
 */
template <typename T>
double set_default_device(const int input_size, const cpu_array<T> input,
                        gpu_array<T> &output, T default_value);

/**
 * @brief Helper function reorder a device array based on a sorted index
 * @details The sorted index is obtained normally obtained from the hashing
 * algorithm. If the topology of the points changes, this function is useful to
 * reorder the array to ensure coalesced memory access. \verbatim
 * embed:rst:leading-asterisk Example usage
 *
 *     .. code-block:: cpp
 *
 *         #include "pyroclastmpm/common/helper.h"
 *
 *         // Reorders an array of points based on a sorted index
 *         reorder_device_array<Vectorr>(positions, sorted_index);
 *
 *
 * \endverbatim
 * @tparam T data type (float, double, int, Matrixr, etc.)
 * @param output array to be reordered
 * @param sorted_index array of sorted indices
 */
template <typename T>
void reorder_device_array(gpu_array<T> &output, gpu_array<int> sorted_index);

/**
 * @brief Helper function to print a device or host array
 * @details This function is useful for debugging purposes
 *
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *         #include "pyroclastmpm/common/helper.h"
 *
 *         // Prints matrices in a formatted way
 *         print_array<Matrixr>(matrices);
 *
 *
 * \endverbatim
 * @tparam T data type (float, double, int, Matrixr, etc.)
 * @param input array to be printed
 */
template <typename T> void print_array(const cpu_array<T> input);

} // namespace pyroclastmpm