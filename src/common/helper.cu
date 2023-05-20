#include <thrust/gather.h>
#include "pyroclastmpm/common/helper.cuh"

namespace pyroclastmpm
{

  template <typename T>
  void set_default_device(const int input_size,
                          const cpu_array<T> input,
                          gpu_array<T> &output,
                          T default_value)
  {

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
  }

  template <typename T>
  void reorder_device_array(gpu_array<T> &output,
                            gpu_array<int> sorted_index)
  {
    gpu_array<T> device_temp = output;
    thrust::gather(sorted_index.begin(), sorted_index.end(), device_temp.begin(),
                   output.begin());
  }

  template <typename T>
  void print_array(const cpu_array<T> input)
  {
    std::cout << std::endl;
    for (auto &out : input)
    {

      if constexpr (std::is_same_v<T, Matrixr> || std::is_same_v<T, Vectorr> || std::is_same_v<T, Matrix3r> || std::is_same_v<T, Vectori>)
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

  template void print_array<bool>(const cpu_array<bool> input);

  template void set_default_device<bool>(const int input_size,
                                         const cpu_array<bool> input,
                                         gpu_array<bool> &output,
                                         bool default_value);

  template void reorder_device_array<bool>(gpu_array<bool> &output,
                                           gpu_array<int> sorted_index);

  template void print_array<uint8_t>(const cpu_array<uint8_t> input);

  template void set_default_device<uint8_t>(const int input_size,
                                            const cpu_array<uint8_t> input,
                                            gpu_array<uint8_t> &output,
                                            uint8_t default_value);

  template void reorder_device_array<uint8_t>(gpu_array<uint8_t> &output,
                                              gpu_array<int> sorted_index);

  template void print_array<unsigned int>(const cpu_array<unsigned int> input);

  template void set_default_device<unsigned int>(const int input_size,
                                                 const cpu_array<unsigned int> input,
                                                 gpu_array<unsigned int> &output,
                                                 unsigned int default_value);

  template void reorder_device_array<unsigned int>(gpu_array<unsigned int> &output,
                                                   gpu_array<int> sorted_index);

  template void print_array<int>(const cpu_array<int> input);

  template void set_default_device<int>(const int input_size,
                                        const cpu_array<int> input,
                                        gpu_array<int> &output,
                                        int default_value);

  template void reorder_device_array<int>(gpu_array<int> &output,
                                          gpu_array<int> sorted_index);

  template void print_array<Real>(const cpu_array<Real> input);

  template void set_default_device<Real>(const int input_size,
                                         const cpu_array<Real> input,
                                         gpu_array<Real> &output,
                                         Real default_value);

  template void reorder_device_array<Real>(gpu_array<Real> &output,
                                           gpu_array<int> sorted_index);

  template void print_array<Matrixr>(const cpu_array<Matrixr> input);

  template void set_default_device<Matrixr>(const int input_size,
                                            const cpu_array<Matrixr> input,
                                            gpu_array<Matrixr> &output,
                                            Matrixr default_value);

  template void reorder_device_array<Matrixr>(gpu_array<Matrixr> &output,
                                              gpu_array<int> sorted_index);

  template void print_array<Vectori>(const cpu_array<Vectori> input);

  template void set_default_device<Vectori>(const int input_size,
                                            const cpu_array<Vectori> input,
                                            gpu_array<Vectori> &output,
                                            Vectori default_value);

  template void reorder_device_array<Vectori>(gpu_array<Vectori> &output,
                                              gpu_array<int> sorted_index);

// Since Matri3r is a typedef of Matrixr is same as Vector (explicitly initiated )
#if DIM != 1
  template void set_default_device<Vectorr>(const int input_size,
                                            const cpu_array<Vectorr> input,
                                            gpu_array<Vectorr> &output,
                                            Vectorr default_value);

  template void reorder_device_array<Vectorr>(gpu_array<Vectorr> &output,
                                              gpu_array<int> sorted_index);

  template void print_array<Vectorr>(const cpu_array<Vectorr> input);
#endif

// Since Matri3r is the same as Matrixr (explicitly initiated )
#if DIM != 3
  template void set_default_device<Matrix3r>(const int input_size,
                                             const cpu_array<Matrix3r> input,
                                             gpu_array<Matrix3r> &output,
                                             Matrix3r default_value);

  template void reorder_device_array<Matrix3r>(gpu_array<Matrix3r> &output,
                                               gpu_array<int> sorted_index);

  template void print_array<Matrix3r>(const cpu_array<Matrix3r> input);
#endif

  // void print_array(const thrust::host_vector<Real> input) {
  //   std::cout << std::endl;
  //   for (auto& out : input) {
  //     std::cout << out << ", ";
  //   }
  //   std::cout << std::endl;
  // }

  // void print_array(const thrust::host_vector<Vector3r> input) {
  //   std::cout << std::endl;
  //   for (auto& out : input) {
  //     std::cout << "(" << out(0) << ", " << out(1) << ", " << out(2) << ") , ";
  //   }
  //   std::cout << std::endl;
  // }

  // void print_array(const thrust::host_vector<Vector3i> input) {
  //   std::cout << std::endl;
  //   for (auto& out : input) {
  //     std::cout << "(" << out(0) << ", " << out(1) << ", " << out(2) << ") , ";
  //   }
  //   std::cout << std::endl;
  // }

  // void print_array(const thrust::host_vector<Matrix3r> input) {
  //   std::cout << std::endl;
  //   for (auto& out : input) {
  //     // std::cout << "(" << out(0) << ", " << out(1) << ", " << out(2) << ") , ";
  //     std::cout << out << ", ";
  //   }
  //   std::cout << std::endl;
  // }

  // // write a test to see if an array returns {1,2,3,4,5,6,7,8,9,10}

  // // // get deviatoric invariant of a stress tensor
  // // void get_deviatoric_invariant(const thrust::host_vector<Matrix3r> input,
  // //                               thrust::host_vector<Real>& output) {
  // //   for (int i = 0; i < input.size(); i++) {
  // //     Matrix3r stress = input[i];
  // //     Real p = (stress(0, 0) + stress(1, 1) + stress(2, 2)) / 3.0;
  // //     Real s = (stress - p * Matrix3r::Identity()).norm();
  // //     output[i] = s;
  // //   }
  // // }

} // namespace pyroclastmpm
