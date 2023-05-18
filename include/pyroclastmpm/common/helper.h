#pragma once

#include "pyroclastmpm/common/types_common.h"


namespace pyroclastmpm
{

    template <typename T>
    void set_default_device(const int input_size,
                            const cpu_array<T> input,
                            gpu_array<T> &output,
                            T default_value);

    template <typename T>
    void reorder_device_array(gpu_array<T> &output,
                              gpu_array<int> sorted_index);

    template <typename T>
    void print_array(const cpu_array<T> input);

} // namespace pyroclastmpm