#pragma once

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/solver/musl/musl.cuh"
#include "pyroclastmpm/solver/tlmpm/tlmpm_kernels.cuh"
#include "pyroclastmpm/common/helper.cuh"

namespace pyroclastmpm
{

  /*! @brief Total Lagrangian MPM Solver*/
  
  struct TLMPM : MUSL
  {
    public:
    // using Solver::Solver;
    /**
     * @brief Construct a new Solver object
     *
     * @param _particles particles container
     * @param _nodes nodes container
     * @param _boundaryconditions a list of boundary conditions to be applied
     * @param _materials a list of materials to be applied
     */
    explicit TLMPM(
        ParticlesContainer _particles,
        NodesContainer _nodes,
        cpu_array<MaterialType> _materials =
            cpu_array<MaterialType>(),
        cpu_array<BoundaryConditionType> _boundaryconditions =
            cpu_array<BoundaryConditionType>(),
        Real _alpha = 0.99);

    gpu_array<Matrix3r> stresses_pk1_gpu;
  
    /*! @brief Grid to particle update */
    void G2P();

    /*!
     * @brief  Solver one iteration
     */
    void solve() override;

    /*!
     * @brief reset temporary arrays
     */
    void reset() override;

    void P2G();

    void CauchyStressToPK1Stress();

    void G2P_double_mapping();
  };

} // namespace pyroclastmpm
