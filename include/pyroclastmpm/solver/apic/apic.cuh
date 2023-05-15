#pragma once

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/solver/apic/apic_kernels.cuh"
#include "pyroclastmpm/solver/solver.cuh"
namespace pyroclastmpm {

/*! @brief Update Stress Last (USL) solver */
struct APIC : Solver {
 public:
  using Solver::Solver;

  // APIC(std::shared_ptr<ParticlesContainer> _particles_ptr = NULL,
  //               std::shared_ptr<NodesContainer> _nodes_ptr = NULL,
  //               thrust::host_vector<BoundaryCondition*> _boundaryconditions =
  //                   thrust::host_vector<BoundaryCondition*>());
  APIC(ParticlesContainer _particles,
       NodesContainer _nodes,
       thrust::host_vector<MaterialType> _materials =
           thrust::host_vector<MaterialType>(),
       thrust::host_vector<BoundaryConditionType> _boundaryconditions =
           thrust::host_vector<BoundaryConditionType>());
  // FUNCTIONS
  /*!
   * @brief Particle to grid update
   */
  void P2G();

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

  // VARIABLES

  /*! @brief particles' velocity gradients */
  // thrust::device_vector<Matrix3r> Bp_gpu;

  Matrix3r Wp_inverse;
};

}  // namespace pyroclastmpm
