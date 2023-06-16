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
 * @file global_settings.h
 * @author Retief Lubbe (r.lubbe@utwente.nl)
 * @brief This header file contains global variables and functions that set
 * the global variables.
 *
 * @version 0.1
 * @date 2023-06-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "pyroclastmpm/common/types_common.h"
#include <string>

namespace pyroclastmpm {

/**
 * @brief Set the global shapefunction type
 *
 * @param _dimension simulation dimension
 */
void set_global_shapefunction(SFType _shapefunction);

/**
 * @brief Set the global time step
 *
 * @param _dt timestep
 */
void set_global_dt(const Real _dt);

/**
 * @brief Set the global output directory string
 *
 * @param output_dir the output directory string
 */
void set_global_output_directory(const std::string _output_dir);

/**
 * @brief Set the initial particles per cell
 *
 * @param _particles_per_cell initial particles per cell
 */
void set_global_particles_per_cell(const int _particles_per_cell);

void set_global_step(const int _step);

/**
 * @brief A master function to set all the globals variables
 *
 * @param _dt timestep
 * @param particles_per_cell initial particles per cell
 * @param _shapefunction shapefunction type
 * @param _output_dir output directory string
 */
void set_globals(const Real _dt, const int particles_per_cell,
                 SFType _shapefunction, const std::string _output_dir);

} // namespace pyroclastmpm