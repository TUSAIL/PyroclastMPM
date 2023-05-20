#ifndef PYROCLASTMPM_MATERIAL_H
#define PYROCLASTMPM_MATERIAL_H

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/particles/particles.cuh"

namespace pyroclastmpm
{

  /*!
   * @brief Material base class
   */
  struct Material
  {
    /*!
     * @brief Default constructor
     */
    Material() = default;

    /*!
     * @brief Constructor for restart files
     */
    Material(Real _density, std::string _name) : density(_density), name(_name){};

    /*!
     * @brief Default destructor
     */
    ~Material() = default;

    /*!
     * @brief Stress update called from a Solver class
     * @param particles_ref Particle references
     * @param mat_id material id
     */
    virtual void stress_update(ParticlesContainer &particles_ref, int mat_id){};

    /**
     * @brief calculate material time step
     * @param cell_size
     * @param factor
     * @return Real
     */
    virtual Real calculate_timestep(Real cell_size, Real factor)
    {
      // printf("Material::calculate_timestep() called\n");
      return 100000;
    };

    Real density = 0.0;

    std::string name = "None";
  };

} // namespace pyroclastmpm

#endif // PYROCLASTMPM_MATERIAL_H