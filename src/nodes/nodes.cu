#include "pyroclastmpm/nodes/nodes.cuh"

namespace pyroclastmpm
{

  NodesContainer::NodesContainer(const Vectorr _node_start,
                                 const Vectorr _node_end,
                                 const Real _node_spacing,
                                 const cpu_array<OutputType> _output_formats)
  {

    output_formats = _output_formats;

    node_start = _node_start;
    node_end = _node_end;
    node_spacing = _node_spacing;

    inv_node_spacing = 1. / node_spacing;

    num_nodes_total = 1;

#if DIM == 3
    num_nodes = Vectori({1, 1, 1});
#elif DIM == 2
    num_nodes = Vectori({1, 1});
#else
    num_nodes = Vectori(1);
#endif

    for (int axis = 0; axis < DIM; axis++)
    {
      num_nodes[axis] =
          (int)((node_end[axis] - node_start[axis]) / node_spacing) + 1;
      num_nodes_total *= num_nodes[axis];
    }
    set_default_device<Vectorr>(num_nodes_total, {}, moments_gpu, Vectorr::Zero());
    set_default_device<Vectorr>(num_nodes_total, {}, moments_nt_gpu, Vectorr::Zero());
    set_default_device<Vectorr>(num_nodes_total, {}, forces_external_gpu, Vectorr::Zero());
    set_default_device<Vectorr>(num_nodes_total, {}, forces_internal_gpu, Vectorr::Zero());
    set_default_device<Vectorr>(num_nodes_total, {}, forces_total_gpu, Vectorr::Zero());
    set_default_device<Real>(num_nodes_total, {}, masses_gpu, 0.);
    set_default_device<Vectori>(num_nodes_total, {}, node_ids_gpu, Vectori::Zero());
    set_default_device<Vectori>(num_nodes_total, {}, node_types_gpu, Vectori::Zero()); // TODO make this Vectoru8 (Vector of 8bit unsigned ints)

    gpuErrchk(cudaDeviceSynchronize());

    reset();

    launch_config.tpb = dim3(int((num_nodes_total) / BLOCKSIZE) + 1, 1, 1);
    launch_config.bpg = dim3(BLOCKSIZE, 1, 1);

#if DIM == 3
    launch_config_map.tpb = dim3(BLOCKSIZE / pow(2, DIM), BLOCKSIZE / pow(2, DIM), BLOCKSIZE / pow(2, DIM));
    launch_config_map.bpg = dim3((num_nodes(0) / launch_config_map.tpb.x) + 1,
                                 (num_nodes(1) / launch_config_map.tpb.y) + 1,
                                 (num_nodes(2) / launch_config_map.tpb.z) + 1);
#elif DIM == 2
    launch_config_map.tpb = dim3(BLOCKSIZE / pow(2, DIM), BLOCKSIZE / pow(2, DIM), 1);
    launch_config_map.bpg = dim3((num_nodes(0) / launch_config_map.tpb.x) + 1,
                                 (num_nodes(1) / launch_config_map.tpb.y) + 1, 1);
#else
    launch_config_map.tpb = dim3(BLOCKSIZE / pow(2, DIM), 1, 1);
    launch_config_map.bpg = dim3((num_nodes(0) / launch_config_map.tpb.x) + 1, 1, 1);
#endif

    KERNEL_GIVE_NODE_IDS<<<launch_config_map.tpb, launch_config_map.bpg>>>(
        thrust::raw_pointer_cast(node_ids_gpu.data()), num_nodes);

    gpuErrchk(cudaDeviceSynchronize());

    KERNEL_SET_NODE_TYPES<<<launch_config_map.tpb, launch_config_map.bpg>>>(
        thrust::raw_pointer_cast(node_types_gpu.data()), num_nodes);

    gpuErrchk(cudaDeviceSynchronize());
  }

  NodesContainer::~NodesContainer() {}

  void NodesContainer::reset()
  {
    thrust::fill(thrust::device, moments_gpu.begin(), moments_gpu.end(),
                 Vectorr::Zero());
    thrust::fill(thrust::device, moments_nt_gpu.begin(), moments_nt_gpu.end(),
                 Vectorr::Zero());
    thrust::fill(thrust::device, forces_external_gpu.begin(),
                 forces_external_gpu.end(), Vectorr::Zero());
    thrust::fill(thrust::device, forces_internal_gpu.begin(),
                 forces_internal_gpu.end(), Vectorr::Zero());
    thrust::fill(thrust::device, forces_total_gpu.begin(), forces_total_gpu.end(),
                 Vectorr::Zero());
    thrust::fill(thrust::device, masses_gpu.begin(), masses_gpu.end(), 0.);
    gpuErrchk(cudaDeviceSynchronize());
  }

  void NodesContainer::integrate()
  {
    KERNEL_INTEGRATE<<<launch_config.tpb, launch_config.bpg>>>(
        thrust::raw_pointer_cast(moments_nt_gpu.data()),
        thrust::raw_pointer_cast(forces_total_gpu.data()),
        thrust::raw_pointer_cast(forces_external_gpu.data()),
        thrust::raw_pointer_cast(forces_internal_gpu.data()),
        thrust::raw_pointer_cast(moments_gpu.data()),
        thrust::raw_pointer_cast(masses_gpu.data()), num_nodes_total);
    gpuErrchk(cudaDeviceSynchronize());
  }

  thrust::device_vector<Vectorr> NodesContainer::give_node_coords()
  {
    gpu_array<Vectorr> node_coords_gpu;

    set_default_device<Vectorr>(num_nodes_total, {}, node_coords_gpu, Vectorr::Zero());

    KERNEL_GIVE_NODE_COORDS<<<launch_config_map.tpb, launch_config_map.bpg>>>(
        thrust::raw_pointer_cast(node_coords_gpu.data()), inv_node_spacing,
        num_nodes);

    cpu_array<Vectorr> node_coords;

    node_coords = node_coords_gpu;

    return node_coords;
  }

  std::vector<Vectorr> NodesContainer::give_node_coords_stl()
  {

    gpu_array<Vectorr> node_coords_gpu;

    set_default_device<Vectorr>(num_nodes_total, {}, node_coords_gpu, Vectorr::Zero());

    KERNEL_GIVE_NODE_COORDS<<<launch_config_map.tpb, launch_config_map.bpg>>>(
        thrust::raw_pointer_cast(node_coords_gpu.data()), inv_node_spacing,
        num_nodes);

    return std::vector<Vectorr>(node_coords_gpu.begin(), node_coords_gpu.end());
  }

  void NodesContainer::output_vtk()
  {

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

    cpu_array<Vectorr> positions_cpu = give_node_coords();
    cpu_array<Vectorr> moments_cpu = moments_gpu;
    cpu_array<Vectorr> moments_nt_cpu = moments_nt_gpu;
    cpu_array<Vectorr> forces_external_cpu = forces_external_gpu;
    cpu_array<Vectorr> forces_internal_cpu = forces_internal_gpu;
    cpu_array<Vectorr> forces_total_cpu = forces_total_gpu;
    cpu_array<Real> masses_cpu = masses_gpu;

    set_vtk_points(positions_cpu, polydata);
    set_vtk_pointdata<Vectorr>(moments_cpu, polydata, "Moments");
    set_vtk_pointdata<Vectorr>(moments_nt_cpu, polydata, "MomentsNT");
    set_vtk_pointdata<Vectorr>(forces_external_cpu, polydata, "ForcesExternal");
    set_vtk_pointdata<Vectorr>(forces_internal_cpu, polydata, "ForcesInternal");
    set_vtk_pointdata<Vectorr>(forces_total_cpu, polydata, "ForcesTotal");
    set_vtk_pointdata<Real>(masses_cpu, polydata, "Mass");

    // loop over output_formats
    for (auto format : output_formats)
    {
      write_vtk_polydata(polydata, "nodes", format);
    }
  }
} // namespace pyroclastmpm