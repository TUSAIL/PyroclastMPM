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
 * @file nodes.cpp
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Implementation of background grid in MPM
 * @details Main purpose of the background grid is to solve
 * governing equations of the MPM method, and temporarily store
 * nodal quantities.
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */
#include "pyroclastmpm/nodes/nodes.h"

#include "nodes_inline.h"

#include "spdlog/spdlog.h"

namespace pyroclastmpm
{

  /// @brief Shape function used to calculate node type
  extern const SFType shape_function_cpu;
  extern const int global_step_cpu;

  ///@brief Construct a new Nodes Container object
  ///@param _node_start Origin of where nodes will be generated
  ///@param _node_end  End where nodes will be generated
  ///@param _node_spacing Cell size of the background grid
  NodesContainer::NodesContainer(const Vectorr _node_start,
                                 const Vectorr _node_end,
                                 const Real _node_spacing)
      : grid(_node_start, _node_end, _node_spacing)
  {

    total_memory_mb += set_default_device<Vectorr>(grid.num_cells_total, {}, moments_gpu,
                                                   Vectorr::Zero());
    total_memory_mb += set_default_device<Vectorr>(grid.num_cells_total, {}, moments_nt_gpu,
                                                   Vectorr::Zero());

    total_memory_mb += set_default_device<Real>(grid.num_cells_total, {}, masses_gpu, 0.);
    total_memory_mb += set_default_device<Vectori>(grid.num_cells_total, {}, node_ids_gpu,
                                                   Vectori::Zero());
    total_memory_mb += set_default_device<Vectori>(grid.num_cells_total, {}, node_types_gpu,
                                                   Vectori::Zero());
    reset();

    calculate_bin_ids();

    calculate_bin_types();

#ifdef CUDA_ENABLED
    launch_config = GPULaunchConfig(grid.num_cells_total);
#endif
    spdlog::info("[Nodes] Number of node: {}", grid.num_cells_total);
    spdlog::info("[Nodes] Total memory allocated: {:2f} MB", total_memory_mb);
    spdlog::info("[Nodes] Memory allocated per node: {:4f} MB", total_memory_mb / grid.num_cells_total);
  }

  ///@brief Calculate the cartesian hash of the nodes (idx,idy,idz)
  ///@details the cartesian hash is calculated using a similar hashmap
  /// as done in the SpatialPartition class. This can be done on the CPU
  /// as it is only done once at the start of the simulation.
  void NodesContainer::calculate_bin_ids()
  {

    cpu_array<Vectori> node_ids_cpu = node_ids_gpu;

    // TODO replace with macro and make less verbose
#if DIM == 1
    for (int xi = 0; xi < grid.num_cells(0); xi++)
    {
      int index = xi;
      node_ids_cpu[index] = Vectori(xi);
    }
#endif

#if DIM == 2
    for (int xi = 0; xi < grid.num_cells(0); xi++)
    {
      for (int yi = 0; yi < grid.num_cells(1); yi++)
      {
        int index = xi + yi * grid.num_cells(0);
        node_ids_cpu[index] = Vectori({xi, yi});
      }
    }
#endif

#if DIM == 3
    for (int xi = 0; xi < grid.num_cells(0); xi++)
    {
      for (int yi = 0; yi < grid.num_cells(1); yi++)
      {
        for (int zi = 0; zi < grid.num_cells(2); zi++)
        {
          int index = xi + yi * grid.num_cells(0) +
                      zi * grid.num_cells(0) * grid.num_cells(1);
          node_ids_cpu[index] = Vectori({xi, yi, zi});
        }
      }
    }
#endif
    node_ids_gpu = node_ids_cpu;
  }

  /// @brief Calculate the node types (boundary, right, left, middle)
  /// @details The node type is important for boundary conditions and
  /// for the calculation of the shape function.This function is only
  /// necessary for higher order shape functions
  void NodesContainer::calculate_bin_types()
  {
    cpu_array<Vectori> node_ids_cpu = node_ids_gpu;
    cpu_array<Vectori> node_types_cpu = node_types_gpu;

    if (shape_function_cpu != CubicShapeFunction)
    {
      return;
    }
    // TODO implement for other higher order shape functions (quadratic)
    for (int index = 0; index < grid.num_cells_total; index++)
    {
      for (int axis = 0; axis < DIM; axis++)
      {
        if ((node_ids_cpu[index][axis] == 0) ||
            (node_ids_cpu[index][axis] == grid.num_cells[axis] - 1))
        {
          // Cell at boundary
          node_types_cpu[index][axis] = 1;
        }
        else if (node_ids_cpu[index][axis] == 1)
        {
          // Cell right of boundary
          node_types_cpu[index][axis] = 2;
        }
        else if (node_ids_cpu[index][axis] == node_ids_cpu[index][axis] - 2)
        {
          // Cell left of boundary
          node_types_cpu[index][axis] = 4;
        }
        else
        {
          node_types_cpu[index][axis] = 3;
        }
      }

      node_types_gpu = node_types_cpu;
    }
  }

  /// @brief Set the output formats ("vtk", "csv", "obj")
  void NodesContainer::set_output_formats(
      const std::vector<std::string> &_output_formats)
  {
    output_formats = _output_formats;
  }

  /// @brief Resets arrays of the background grid
  void NodesContainer::reset()
  {
    thrust::fill(moments_gpu.begin(), moments_gpu.end(), Vectorr::Zero());
    thrust::fill(moments_nt_gpu.begin(), moments_nt_gpu.end(), Vectorr::Zero());
    thrust::fill(masses_gpu.begin(), masses_gpu.end(), 0.);
  }

  /// @brief Get the node coordinates as a gpu array
  gpu_array<Vectorr> NodesContainer::give_node_coords() const
  {
    gpu_array<Vectorr> node_coords_cpu;
    node_coords_cpu.resize(grid.num_cells_total);
    cpu_array<Vectori> node_ids_cpu = node_ids_gpu;
    for (size_t i = 0; i < grid.num_cells_total; i++)
    {
      node_coords_cpu[i] =
          grid.origin + node_ids_cpu[i].cast<Real>() * grid.cell_size;
    }
    gpu_array<Vectorr> node_coords_gpu = node_coords_cpu;
    return node_coords_gpu;
  }

  /// @brief Get the node coordinates as a stl vector
  std::vector<Vectorr> NodesContainer::give_node_coords_stl() const
  {
    gpu_array<Vectorr> node_coords_gpu = give_node_coords();
    return std::vector<Vectorr>(node_coords_gpu.begin(), node_coords_gpu.end());
  }

  /// @brief Output node data
  /// @details Requires that `set_output_formats` is called first
  void NodesContainer::output_vtk() const
  {

    if (output_formats.empty())
    {
      return;
    }
    if (global_step_cpu > 1)
    { // TODO: Remove
      return;
    }

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

    cpu_array<Vectorr> positions_cpu = give_node_coords();
    cpu_array<Vectorr> moments_cpu = moments_gpu;
    cpu_array<Vectorr> moments_nt_cpu = moments_nt_gpu;

    cpu_array<Real> masses_cpu = masses_gpu;

    set_vtk_points(positions_cpu, polydata);
    set_vtk_pointdata<Vectorr>(moments_cpu, polydata, "Moments");
    set_vtk_pointdata<Vectorr>(moments_nt_cpu, polydata, "MomentsNT");
    set_vtk_pointdata<Real>(masses_cpu, polydata, "Mass");

    // loop over output_formats
    for (const auto &format : output_formats)
    {
      write_vtk_polydata(polydata, "nodes", format);
    }
  }
} // namespace pyroclastmpm