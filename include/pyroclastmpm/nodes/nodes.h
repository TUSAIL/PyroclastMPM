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
 * @file nodes.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief MPM background grid nodes class
 * @details The background grid is implemented here by the NodesContainer class
 * which solves the governing equations.
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#include "pyroclastmpm/common/helper.h"
#include "pyroclastmpm/common/output.h"
#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm {

/**
 * @brief Background grid nodes
 * @details A background grid is implemented here which serves
 * as the main data structure to store nodal quantities (mass, forces etc)
 * and solve the governing equations.
 * \verbatim embed:rst:leading-asterisk
 *     Example usage
 *
 *     .. code-block:: cpp
 *
 *        #include "pyroclastmpm/nodes/nodes.h"
 *
 *        Vectorr min = Vectorr::Zero();

 *        Vectorr max = Vectorr::Ones();
 *
 *        Real nodal_spacing = 0.5;
 *
 *        NodesContainer nodes = NodesContainer(min, max, nodal_spacing);
 *
 *        Vectorr node_coords = nodes.give_node_coords();
 *
 *
 * \endverbatim
 *
 */
class NodesContainer {

private:
  /// @brief Calculates the node ids
  void calculate_bin_ids();

  /// @brief Calculates the node types (i.e boundary or interior)
  void calculate_bin_types();

public:
  /// @brief Construct a new Nodes Container object (default constructor)
  NodesContainer() = default;

  ///@brief Construct a new Nodes Container object
  ///@param _node_start Origin of where nodes will be generated
  ///@param _node_end  End where nodes will be generated
  ///@param _node_spacing Cell size of the background grid
  NodesContainer(const Vectorr _node_start, const Vectorr _node_end,
                 const Real _node_spacing);

  /// @brief Destroy the NodesContainer object
  ~NodesContainer() = default;

  /// @brief Resets arrays of the background grid
  void reset();

  /// @brief Give the coordinates of the nodes as a flattened array
  gpu_array<Vectorr> give_node_coords() const;

  /// @brief Give the coordinates of the nodes as a flattened array (as stl)
  std::vector<Vectorr> give_node_coords_stl() const;

  /// @brief Integrate the nodal forces to the momentu
  void integrate();

  /// @brief Output particle data ("vtk", "csv", "obj")
  /// @details Requires that `set_output_formats` is called first
  void output_vtk() const;

  /// @brief Set output format of the nodal quantities
  void set_output_formats(const std::vector<std::string> &_output_formats);

  /// @brief Current moment of the nodes
  gpu_array<Vectorr> moments_gpu;

  /// @brief Forward moment of the nodes (related to the USL integration)
  gpu_array<Vectorr> moments_nt_gpu;

  /// @brief External forces of the nodes (i.e external loads or gravity)
  gpu_array<Vectorr> forces_external_gpu;

  /// @brief Internal forces of the nodes (from particle stresses)
  gpu_array<Vectorr> forces_internal_gpu;

  /// @brief Total forces of the nodes (F_INT + F_EXT)
  gpu_array<Vectorr> forces_total_gpu;

  /// @brief Masses of the nodes
  gpu_array<Real> masses_gpu;

  /// @brief Ids of the nodes at a bin level (i,j,k)
  gpu_array<Vectori> node_ids_gpu;

  /// @brief Types of the nodes (boundary or interior)
  gpu_array<Vectori> node_types_gpu;

  /// @brief Information about the grid (number of cells, cell size etc)
  Grid grid = Grid();

#ifdef CUDA_ENABLED
  /// @brief CUDA GPU launch configuration
  GPULaunchConfig launch_config;
#endif

  /// @brief Output formats for the nodes
  std::vector<std::string> output_formats;

  /// @brief Small mass value to prevent division by zero
  Real small_mass_cutoff = (Real)1.0e-6;
};

} // namespace pyroclastmpm