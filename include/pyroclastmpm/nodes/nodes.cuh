#pragma once
// #include <thrust/execution_policy.h>

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/common/helper.cuh"
// #include "pyroclastmpm/common/output.cuh"
// #include "pyroclastmpm/nodes/nodes_kernels.cuh"

namespace pyroclastmpm
{

  /**
   * @brief The nodes container contains information on the background nodes
   *
   */
  class NodesContainer
  {
  public:
    // Tell the compiler to do what it would have if we didn't define a ctor:
    NodesContainer() = default;

    /**
     * @brief Construct a new Nodes Container object
     *
     * @param _node_start origin of where nodes will be generated
     * @param _node_end  end where nodes will be generated
     * @param _node_spacing cell size of the backgrund grid
     */
    NodesContainer(const Vectorr _node_start,
                   const Vectorr _node_end,
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

    // GPULaunchConfig launch_config;

    // temp for mapping grid to ids
    // GPULaunchConfig launch_config_map;


    cpu_array<OutputType> output_formats;
  };

} // namespace pyroclastmpm