#include "pyroclastmpm/common/global_settings.cuh"

namespace pyroclastmpm
{

  char output_directory_cpu[256];

#ifdef CUDA_ENABLED
  __constant__ SFType shape_function_gpu = LinearShapeFunction;

  __constant__ int num_surround_nodes_gpu;
  __constant__ Real dt_gpu;
  __constant__ Real inv_cell_size_gpu;
  __constant__ int global_step_gpu = 0;
  __constant__ int window_size_gpu;
  __constant__ int particles_per_cell_gpu;
  __constant__ int forward_window_gpu[64][3];
  __constant__ int backward_window_gpu[64][3];
#endif

  SFType shape_function_cpu = LinearShapeFunction;
  int num_surround_nodes_cpu;
  Real dt_cpu;
  Real inv_cell_size_cpu;
  int global_step_cpu = 0;
  int window_size_cpu;

  int particles_per_cell_cpu;

  const int linear_forward_window_1d[64][3] = {{0, 0, 0}, {1, 0, 0}};

  const int linear_forward_window_2d[64][3] = {{0, 0, 0},
                                               {1, 0, 0},
                                               {0, 1, 0},
                                               {1, 1, 0}};

  const int linear_forward_window_3d[64][3] = {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 1, 1}};

  const int linear_backward_window_1d[64][3] = {{0, 0, 0}, {-1, 0, 0}};

  const int linear_backward_window_2d[64][3] = {{0, 0, 0},
                                                {-1, 0, 0},
                                                {0, -1, 0},
                                                {-1, -1, 0}};

  const int linear_backward_window_3d[64][3] = {
      {0, 0, 0}, {0, 0, -1}, {-1, 0, 0}, {-1, 0, -1}, {0, -1, 0}, {0, -1, -1}, {-1, -1, 0}, {-1, -1, -1}};

  const int quadratic_forward_window_1d[64][3] = {{-1, 0, 0},
                                                  {0, 0, 0},
                                                  {1, 0, 0},
                                                  {2, 0, 0}};

  const int quadratic_forward_window_2d[64][3] = {
      {-1, -1, 0}, {0, -1, 0}, {1, -1, 0}, {2, -1, 0}, {-1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {-1, 1, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {-1, 2, 0}, {0, 2, 0}, {1, 2, 0}, {2, 2, 0}};

  const int quadratic_forward_window_3d[64][3] = {
      {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, -1, 2}, {0, -1, -1}, {0, -1, 0}, {0, -1, 1}, {0, -1, 2}, {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, -1, 2}, {2, -1, -1}, {2, -1, 0}, {2, -1, 1}, {2, -1, 2}, {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1}, {-1, 0, 2}, {0, 0, -1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {2, 0, -1}, {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1}, {-1, 1, 2}, {0, 1, -1}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {1, 1, -1}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {2, 1, -1}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {-1, 2, -1}, {-1, 2, 0}, {-1, 2, 1}, {-1, 2, 2}, {0, 2, -1}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2}, {1, 2, -1}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2}, {2, 2, -1}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2}};

  const int quadratic_backward_window_1d[64][3] = {{1, 0, 0},
                                                   {0, 0, 0},
                                                   {-1, 0, 0},
                                                   {-2, 0, 0}};

  const int quadratic_backward_window_2d[64][3] = {
      {1, 1, 0}, {0, 1, 0}, {-1, 1, 0}, {-2, 1, 0}, {1, 0, 0}, {0, 0, 0}, {-1, 0, 0}, {-2, 0, 0}, {1, -1, 0}, {0, -1, 0}, {-1, -1, 0}, {-2, -1, 0}, {1, -2, 0}, {0, -2, 0}, {-1, -2, 0}, {-2, -2, 0}};

  const int quadratic_backward_window_3d[64][3] = {
      {1, 1, 1}, {1, 1, 0}, {1, 1, -1}, {1, 1, -2}, {0, 1, 1}, {0, 1, 0}, {0, 1, -1}, {0, 1, -2}, {-1, 1, 1}, {-1, 1, 0}, {-1, 1, -1}, {-1, 1, -2}, {-2, 1, 1}, {-2, 1, 0}, {-2, 1, -1}, {-2, 1, -2}, {1, 0, 1}, {1, 0, 0}, {1, 0, -1}, {1, 0, -2}, {0, 0, 1}, {0, 0, 0}, {0, 0, -1}, {0, 0, -2}, {-1, 0, 1}, {-1, 0, 0}, {-1, 0, -1}, {-1, 0, -2}, {-2, 0, 1}, {-2, 0, 0}, {-2, 0, -1}, {-2, 0, -2}, {1, -1, 1}, {1, -1, 0}, {1, -1, -1}, {1, -1, -2}, {0, -1, 1}, {0, -1, 0}, {0, -1, -1}, {0, -1, -2}, {-1, -1, 1}, {-1, -1, 0}, {-1, -1, -1}, {-1, -1, -2}, {-2, -1, 1}, {-2, -1, 0}, {-2, -1, -1}, {-2, -1, -2}, {1, -2, 1}, {1, -2, 0}, {1, -2, -1}, {1, -2, -2}, {0, -2, 1}, {0, -2, 0}, {0, -2, -1}, {0, -2, -2}, {-1, -2, 1}, {-1, -2, 0}, {-1, -2, -1}, {-1, -2, -2}, {-2, -2, 1}, {-2, -2, 0}, {-2, -2, -1}, {-2, -2, -2}};

  void set_globals(const Real _dt,
                   const int particles_per_cell,
                   SFType _shapefunction,
                   const std::string _output_dir)
  {
    set_global_dt(_dt);
    set_global_shapefunction(_shapefunction);
    set_global_output_directory(_output_dir);
    set_global_particles_per_cell(particles_per_cell);
  }

  void set_global_particles_per_cell(const int _particles_per_cell)
  {
    particles_per_cell_cpu = _particles_per_cell;
#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(particles_per_cell_gpu, &(_particles_per_cell), sizeof(int), 0);
#endif
  }

  void set_global_output_directory(const std::string _output_dir)
  {
    std::strcpy(output_directory_cpu, _output_dir.c_str());
  };

  void set_global_dt(const Real _dt)
  {
    dt_cpu = _dt;
#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(dt_gpu, &(_dt), sizeof(Real), 0);
#endif
  };

  void set_global_step(const int _step)
  {
    global_step_cpu = _step;
#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(global_step_gpu, &(_step), sizeof(int), 0);
#endif
  };

  void set_global_shapefunction(SFType _shapefunction)
  {
    // TODO this can be constant defined in the kernel
    if (_shapefunction == LinearShapeFunction)
    {
      window_size_cpu = 2;
      num_surround_nodes_cpu = pow(window_size_cpu, DIM);
      shape_function_cpu = LinearShapeFunction;
#ifdef CUDA_ENABLED
      cudaMemcpyToSymbol(shape_function_gpu, &(shape_function_cpu), sizeof(int), 0);
#if DIM == 1
      cudaMemcpyToSymbol(forward_window_gpu, linear_forward_window_1d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, linear_backward_window_1d,
                         64 * 3 * sizeof(int));
#elif DIM == 2
      cudaMemcpyToSymbol(forward_window_gpu, linear_forward_window_2d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, linear_backward_window_2d,
                         64 * 3 * sizeof(int));
#else
      cudaMemcpyToSymbol(forward_window_gpu, linear_forward_window_3d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, linear_backward_window_3d,
                         64 * 3 * sizeof(int));
#endif
#endif
    }
    else if (_shapefunction == QuadraticShapeFunction)
    {
      window_size_cpu = 4;
      num_surround_nodes_cpu = pow(window_size_cpu, DIM);
      shape_function_cpu = QuadraticShapeFunction;
#ifdef CUDA_ENABLED
      cudaMemcpyToSymbol(shape_function_gpu, &(shape_function_cpu), sizeof(int),
                         0);

#if DIM == 1
      cudaMemcpyToSymbol(forward_window_gpu, quadratic_forward_window_1d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, quadratic_backward_window_1d,
                         64 * 3 * sizeof(int));
#elif DIM == 2
      cudaMemcpyToSymbol(forward_window_gpu, quadratic_forward_window_2d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, quadratic_backward_window_2d,
                         64 * 3 * sizeof(int));
#else
      cudaMemcpyToSymbol(forward_window_gpu, quadratic_forward_window_3d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, quadratic_backward_window_3d,
                         64 * 3 * sizeof(int));
#endif
#endif
    }
    else if (_shapefunction == CubicShapeFunction)
    {
      window_size_cpu = 4;
      num_surround_nodes_cpu = pow(window_size_cpu, DIM);
      shape_function_cpu = CubicShapeFunction;

#ifdef CUDA_ENABLED
      cudaMemcpyToSymbol(shape_function_gpu, &(shape_function_cpu), sizeof(int),
                         0);
#if DIM == 1
      cudaMemcpyToSymbol(forward_window_gpu, quadratic_forward_window_1d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, quadratic_backward_window_1d,
                         64 * 3 * sizeof(int));
#elif DIM == 2
      cudaMemcpyToSymbol(forward_window_gpu, quadratic_forward_window_2d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, quadratic_backward_window_2d,
                         64 * 3 * sizeof(int));
#else
      cudaMemcpyToSymbol(forward_window_gpu, quadratic_forward_window_3d,
                         64 * 3 * sizeof(int));
      cudaMemcpyToSymbol(backward_window_gpu, quadratic_backward_window_3d,
                         64 * 3 * sizeof(int));
#endif

#endif
    }
#ifdef CUDA_ENABLED
    cudaMemcpyToSymbol(window_size_gpu, &(window_size_cpu), sizeof(int), 0);

    cudaMemcpyToSymbol(num_surround_nodes_gpu, &(num_surround_nodes_cpu),
                       sizeof(int), 0);
#endif
  };



}; // namespace pyroclastmpmma