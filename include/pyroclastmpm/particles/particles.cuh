#pragma once

// #include <thrust/execution_policy.h>
// #include <thrust/unique.h>
// #include <thrust/remove.h>
#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/common/helper.cuh"
#include "pyroclastmpm/common/output.cuh"
// #include "pyroclastmpm/particles/particles_kernels.cuh"
#include "pyroclastmpm/spatialpartition/spatialpartition.cuh"

namespace pyroclastmpm
{
  /*!
   * @brief Particles container class
   */
  class ParticlesContainer
  {
  public:
    // Tell the compiler to do what it would have if we didn't define actor:
    ParticlesContainer() = default;

    /*!
     * @brief Constructs Particle container class
     * @param _positions particle positions
     * @param _velocities particle velocities
     * @param _colors particle types (optional)
     * @param _stresses particle stresses (optional)
     * @param _masses particle masses (optional)
     * @param _volumes particle volumes (optional)
     */
    ParticlesContainer(
        const cpu_array<Vectorr> _positions,
        const cpu_array<Vectorr> _velocities = {},
        const cpu_array<uint8_t> _colors = {},
        const cpu_array<Matrix3r> _stresses = {},
        const cpu_array<Real> _masses = {},
        const cpu_array<Real> _volumes = {},
        const cpu_array<OutputType> _output_formats = {});

    /**
     * @brief Resets the gpu arrays
     *
     * @param reset_psi if the node/particle shape functions should be reset
     */
    void reset(bool reset_psi = true);

    /*! @brief Reorder Particles arrays */
    void reorder();

    /*!
     * @brief calls the SpatialPartion class to partition particles into a
     * background grid
     */
    void partition();

    /*! @brief output vtk files */
    void output_vtk();

    /*! @brief calculate the particle volumes */
    void calculate_initial_volumes();

    /*! @brief calculate the particle masses */
    void calculate_initial_masses(int mat_id, Real density);

    /**
     * @brief Set the spatial partition of the particles
     *
     * @param start start of the spatial partition domain
     * @param end  end of the spatial partition domain
     * @param spacing cell size of the spatial partition
     */
    void set_spatialpartition(const Vectorr start,
                              const Vectorr end,
                              const Real spacing);

    /*! @brief particles' stresses, we allways store 3x3 matrix for stresses */
    gpu_array<Matrix3r> stresses_gpu;

    /*! @brief particles' velocity gradients */
    gpu_array<Matrixr> velocity_gradient_gpu;

    /*! @brief particles' deformation matrices */
    gpu_array<Matrixr> F_gpu;

    /* TODO is this needed*/
    /*! @brief particles' strain increments, stored only if
     * store_strain_increments=true */
    gpu_array<Matrixr> strain_increments_gpu;

    /*! @brief particles' shape function gradients */
    gpu_array<Vectorr> dpsi_gpu;

    /*! @brief particles' coordinates */
    gpu_array<Vectorr> positions_gpu;

    /*! @brief particles' velocities */
    gpu_array<Vectorr> velocities_gpu;

    /*! @brief particles' velocities */
    gpu_array<Vectorr> forces_external_gpu;

    /*! @brief particles' masses */
    gpu_array<Real> masses_gpu;

    /*! @brief particles' volumes */
    gpu_array<Real> volumes_gpu;

    /*! @brief particles' volumes */
    gpu_array<Real> volumes_original_gpu;

    /*! @brief particles' shape functions */
    gpu_array<Real> psi_gpu;

    /*! @brief particles' densities (updated each step)*/
    gpu_array<Real> densities_gpu;

    /*! * @brief particles' pressure (updated each step) */
    gpu_array<Real> pressures_gpu;

    /*! @brief particles' phases (updated each step) */
    gpu_array<uint8_t> phases_gpu;

    /*! @brief particles' colors (or material type) */
    gpu_array<uint8_t> colors_gpu;

    /*! * @brief granular fluidity d2g */
    gpu_array<Real> logJp_gpu;

    /*! * @brief plastic deformation matrix */
    gpu_array<Matrixr> FP_gpu;

    /*! * @brief shear stress */
    gpu_array<Real> mu_gpu;

    /*! * @brief granular fluidity */
    gpu_array<Real> g_gpu;

    /*! * @brief granular fluidity d2g */
    gpu_array<Real> ddg_gpu;

    /*! @brief spatial partitioning class */
    SpatialPartition spatial;

    /*! @brief array lags */
    bool store_density, store_volume_corrector;

    /*! @brief store pressure */
    bool store_pressure;

    /*! @brief store strain increment */
    bool store_strain_increments;

    /*! @brief store phases (solid 0, fluid 1, gas 2) */
    bool store_phases;

    /*! @brief Total Number of particles */
    int num_particles;

#ifdef CUDA_ENABLED
    GPULaunchConfig launch_config;
#endif

    cpu_array<OutputType> output_formats;

    bool isRestart = false;

    int numColors = 0;
  };

} // namespace pyroclastmpm
