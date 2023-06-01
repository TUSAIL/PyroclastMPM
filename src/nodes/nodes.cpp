#include "pyroclastmpm/nodes/nodes.h"

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#else
extern Real dt_cpu;
#endif

extern SFType shape_function_cpu;

#include "nodes_inline.h"

NodesContainer::NodesContainer(const Vectorr _node_start,
                               const Vectorr _node_end,
                               const Real _node_spacing,
                               const cpu_array<OutputType> _output_formats)
    : node_start(_node_start), node_end(_node_end), node_spacing(_node_spacing),
      output_formats(_output_formats) {
  inv_node_spacing = 1.0 / node_spacing;
  num_nodes_total = 1;
  num_nodes = Vectori::Ones();

  for (int axis = 0; axis < DIM; axis++) {
    num_nodes[axis] =
        (int)((node_end[axis] - node_start[axis]) / node_spacing) + 1;
    num_nodes_total *= num_nodes[axis];
  }
  set_default_device<Vectorr>(num_nodes_total, {}, moments_gpu,
                              Vectorr::Zero());
  set_default_device<Vectorr>(num_nodes_total, {}, moments_nt_gpu,
                              Vectorr::Zero());
  set_default_device<Vectorr>(num_nodes_total, {}, forces_external_gpu,
                              Vectorr::Zero());
  set_default_device<Vectorr>(num_nodes_total, {}, forces_internal_gpu,
                              Vectorr::Zero());
  set_default_device<Vectorr>(num_nodes_total, {}, forces_total_gpu,
                              Vectorr::Zero());
  set_default_device<Real>(num_nodes_total, {}, masses_gpu, 0.);
  set_default_device<Vectori>(num_nodes_total, {}, node_ids_gpu,
                              Vectori::Zero());
  set_default_device<Vectori>(num_nodes_total, {}, node_types_gpu,
                              Vectori::Zero());
  reset();

  // Calculate integer placement of nodes (x,y,z) along the grid
  cpu_array<Vectori> node_ids_cpu = node_ids_gpu;
  cpu_array<Vectori> node_types_cpu = node_types_gpu;
#if DIM == 1
  for (int xi = 0; xi < num_nodes(0); xi++) {
    int index = xi;
    node_ids_cpu[index] = Vectori(xi);
  }
#endif

#if DIM == 2
  for (int xi = 0; xi < num_nodes(0); xi++) {
    for (int yi = 0; yi < num_nodes(1); yi++) {
      int index = xi + yi * num_nodes(0);
      node_ids_cpu[index] = Vectori({xi, yi});
    }
  }
#endif

#if DIM == 3
  for (int xi = 0; xi < num_nodes(0); xi++) {
    for (int yi = 0; yi < num_nodes(1); yi++) {
      for (int zi = 0; zi < num_nodes(2); zi++) {
        int index = xi + yi * num_nodes(0) + zi * num_nodes(0) * num_nodes(1);
        node_ids_cpu[index] = Vectori({xi, yi, zi});
      }
    }
  }
#endif

  for (int index = 0; index < num_nodes_total; index++) {
    for (int axis = 0; axis < DIM; axis++) {
      if (shape_function_cpu == CubicShapeFunction) {
        if ((node_ids_cpu[index][axis] == 0) |
            (node_ids_cpu[index][axis] == num_nodes[axis] - 1)) {
          // Cell at boundary
          node_types_cpu[index][axis] = 1;
        } else if (node_ids_cpu[index][axis] == 1) {
          // Cell right of boundary
          node_types_cpu[index][axis] = 2;
        } else if (node_ids_cpu[index][axis] == node_ids_cpu[index][axis] - 2) {
          // Cell left of boundary
          node_types_cpu[index][axis] = 4;
        } else {
          node_types_cpu[index][axis] = 3;
        }
      }
    }
  }
  node_types_gpu = node_types_cpu;
  node_ids_gpu = node_ids_cpu;

#ifdef CUDA_ENABLED
  launch_config.tpb = dim3(int((num_nodes_total) / BLOCKSIZE) + 1, 1, 1);
  launch_config.bpg = dim3(BLOCKSIZE, 1, 1);
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

NodesContainer::~NodesContainer() {}

void NodesContainer::reset() {
  execution_policy exec;
  thrust::fill(exec, moments_gpu.begin(), moments_gpu.end(), Vectorr::Zero());
  thrust::fill(exec, moments_nt_gpu.begin(), moments_nt_gpu.end(),
               Vectorr::Zero());
  thrust::fill(exec, forces_external_gpu.begin(), forces_external_gpu.end(),
               Vectorr::Zero());
  thrust::fill(exec, forces_internal_gpu.begin(), forces_internal_gpu.end(),
               Vectorr::Zero());
  thrust::fill(exec, forces_total_gpu.begin(), forces_total_gpu.end(),
               Vectorr::Zero());
  thrust::fill(exec, masses_gpu.begin(), masses_gpu.end(), 0.);
}

struct IntegrateFunctor {
  template <typename Tuple>
  __host__ __device__ void operator()(Tuple tuple) const {
    Vectorr &moments_nt = thrust::get<0>(tuple);
    Vectorr &forces_total = thrust::get<1>(tuple);
    const Vectorr &forces_external = thrust::get<2>(tuple);
    const Vectorr &forces_internal = thrust::get<3>(tuple);
    const Vectorr &moments = thrust::get<4>(tuple);
    const Real &mass = thrust::get<5>(tuple);
    if (mass <= 0.000000001) {
      return;
    }
    const Vectorr ftotal = forces_internal + forces_external;
    forces_total = ftotal;
#ifdef CUDA_ENABLED
    moments_nt = moments + ftotal * dt_gpu;
#else
    moments_nt = moments + ftotal * dt_cpu;
#endif
  }
};

void NodesContainer::integrate() {

#ifdef CUDA_ENABLED
  KERNEL_INTEGRATE<<<launch_config.tpb, launch_config.bpg>>>(
      thrust::raw_pointer_cast(moments_nt_gpu.data()),
      thrust::raw_pointer_cast(forces_total_gpu.data()),
      thrust::raw_pointer_cast(forces_external_gpu.data()),
      thrust::raw_pointer_cast(forces_internal_gpu.data()),
      thrust::raw_pointer_cast(moments_gpu.data()),
      thrust::raw_pointer_cast(masses_gpu.data()), num_nodes_total);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int nid = 0; nid < num_nodes_total; nid++) {
    integrate_nodes(moments_nt_gpu.data(), forces_total_gpu.data(),
                    forces_external_gpu.data(), forces_internal_gpu.data(),
                    moments_gpu.data(), masses_gpu.data(), nid);
  }
#endif
}

gpu_array<Vectorr> NodesContainer::give_node_coords() {
  gpu_array<Vectorr> node_coords_cpu;
  node_coords_cpu.resize(num_nodes_total);
  cpu_array<Vectori> node_ids_cpu = node_ids_gpu;
  for (size_t i = 0; i < num_nodes_total; i++) {
    node_coords_cpu[i] =
        node_start + node_ids_cpu[i].cast<Real>() * node_spacing;
  }
  gpu_array<Vectorr> node_coords_gpu = node_coords_cpu;
  return node_coords_gpu;
}

std::vector<Vectorr> NodesContainer::give_node_coords_stl() {
  gpu_array<Vectorr> node_coords_gpu = give_node_coords();
  return std::vector<Vectorr>(node_coords_gpu.begin(), node_coords_gpu.end());
}

void NodesContainer::output_vtk() {

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
  for (auto format : output_formats) {
    write_vtk_polydata(polydata, "nodes", format);
  }
}
} // namespace pyroclastmpm