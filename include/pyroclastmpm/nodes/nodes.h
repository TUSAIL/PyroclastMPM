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

#include "pyroclastmpm/common/helper.h"
#include "pyroclastmpm/common/output.h"
#include "pyroclastmpm/common/types_common.h"

namespace pyroclastmpm {

/**
 * @brief The nodes container contains information on the background nodes
 *
 */
class NodesContainer {
public:
  // Tell the compiler to do what it would have if we didn't define a ctor:
  NodesContainer() = default;

  /**
   * @brief Construct a new Nodes Container object
   *
   * @param _node_start origin of where nodes will be generated
   * @param _node_end  end where nodes will be generated
   * @param _node_spacing cell size of the background grid
   */
  NodesContainer(const Vectorr _node_start, const Vectorr _node_end,
                 const Real _node_spacing,
                 const cpu_array<OutputType> _output_formats = {});

  ~NodesContainer();

  /** @brief Resets the background grid */
  void reset();

  /** @brief Calls CUDA kernel to calculate the nodal coordinates from the hash
   * table */
  gpu_array<Vectorr> give_node_coords();

  // TODO is this function needed?
  /** @brief Calls CUDA kernel to calculate the nodal coordinates from the hash
   * table. Given as an STL output */
  std::vector<Vectorr> give_node_coords_stl();

  /** @brief integrate the nodal forces to the momentum */
  void integrate();

  /** @brief integrate the nodal forces to the momentum */
  void output_vtk();

  // VARIABLES

  /** @brief current moment of the nodes */
  gpu_array<Vectorr> moments_gpu;

  /** @brief Forward moment of the nodes (related to the USL integration) */
  gpu_array<Vectorr> moments_nt_gpu;

  /** @brief External forces of the nodes (i.e external loads or gravity) */
  gpu_array<Vectorr> forces_external_gpu;

  /** @brief Internal forces of the nodes (from particle stresses) */
  gpu_array<Vectorr> forces_internal_gpu;

  /** @brief Total force of the nodes (F_INT + F_EXT)   */
  gpu_array<Vectorr> forces_total_gpu;

  /** @brief Masses of the nodes */
  gpu_array<Real> masses_gpu;

  /** @brief Masses of the nodes */
  gpu_array<Vectori> node_ids_gpu;

  /** @brief Masses of the nodes */
  gpu_array<Vectori> node_types_gpu;

  /** @brief Simulation start domain */
  Vectorr node_start;

  /** @brief Simulation end domain */
  Vectorr node_end;

  /** @brief Grid size of the background grid */
  Real node_spacing;

  /** @brief Number of nodes in the background grid Mx*My*Mz */
  int num_nodes_total;

  /** @brief Number of nodes on each axis */
  Vectori num_nodes;

  /** @brief Inverse grid size of the background grid */
  Real inv_node_spacing;

#ifdef CUDA_ENABLED
  GPULaunchConfig launch_config;
#endif

  cpu_array<OutputType> output_formats;
};

} // namespace pyroclastmpm