#pragma once

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/solver/usl/usl.cuh"
#include "pyroclastmpm/solver/usl/usl_kernels.cuh"
#include "pyroclastmpm/solver/musl/musl_kernels.cuh"

namespace pyroclastmpm
{

  /*!
   * @brief Modified Update Stress Last Solver (MUSL)
   */
  struct MUSL : USL
  {
  public:

    explicit MUSL(
        ParticlesContainer _particles,
        NodesContainer _nodes,
        cpu_array<MaterialType> _materials =
            cpu_array<MaterialType>(),
        cpu_array<BoundaryConditionType> _boundaryconditions =
            cpu_array<BoundaryConditionType>(),
        Real _alpha = 0.99);

    /*!
     * @brief Particle to grid transfer 2
     */
    void P2G_double_mapping();

    /*!
     * @brief Grid to particle transfer 1
     */
    void G2P_double_mapping();

    /*!
     * @brief Grid to particles transfter 2
     */
    void G2P();

    /*!
     * @brief Solver one iteration
     */
    void solve() override;

  };

} // namespace pyroclastmpm
