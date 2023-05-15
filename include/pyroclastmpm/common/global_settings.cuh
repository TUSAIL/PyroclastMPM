#pragma once

#include <string>
#include "pyroclastmpm/common/types_common.cuh"

namespace pyroclastmpm
{
    /**
     * @brief Set the global shapefunction object
     *
     * @param _dimension simulation dimension
     * @param _shapefunction shape function
     */
    void set_global_shapefunction(SFType _shapefunction);

    /**
     * @brief Set the global time step
     *
     * @param _dt timestep
     */
    void set_global_dt(const Real _dt);

    /**
     * @brief Set the global output directory object
     *
     * @param output_dir
     */
    void set_global_output_directory(const std::string _output_dir);

    void set_globals(const Real _dt,
                     const int particles_per_cell,
                     SFType _shapefunction,
                     const std::string _output_dir);

    void set_global_particles_per_cell(const int _particles_per_cell);

    void set_global_step(const int _step);

} // namespace pyroclastmpm
