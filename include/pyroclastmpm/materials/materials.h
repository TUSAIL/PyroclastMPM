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

/**
 * @file materials.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief Material base class
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include "pyroclastmpm/common/types_common.h"
#include "pyroclastmpm/nodes/nodes.h"
#include "pyroclastmpm/particles/particles.h"

namespace pyroclastmpm {

/// @brief This is a base class for every Material
class Material {
public:
  /*!
   * @brief Default constructor
   */
  Material() = default;

  /// @brief Constructor for restart files
  explicit Material(Real _density) : density(_density){};

  /// @brief Default destructor
  virtual ~Material() = default;

  /// @brief Stress update called from a Solver class
  /// @param particles_ref Particle references
  /// @param mat_id material id
  virtual void stress_update(ParticlesContainer &particles_ref, int mat_id){};

  /// @brief Calculate time step wave propagation speed
  /// @param cell_size Fell size of the background grid
  /// @param factor Scaling factor for speed
  /// @return Real a timestep
  virtual Real calculate_timestep(Real cell_size, Real factor) {
    return 100000;
  };

  virtual void output_vtk(NodesContainer &nodes_ref,
                          ParticlesContainer &particles_ref){
      // override this function
  };

  /// @brief material densities (per particle)
  gpu_array<Real> densities_gpu;

  /// @brief material pressure (per particle)
  gpu_array<Real> pressures_gpu;

  /// @brief Initial bulk density
  Real density = 0.0;

  /// @brief Output formats
  std::vector<std::string> output_formats;
};

} // namespace pyroclastmpm