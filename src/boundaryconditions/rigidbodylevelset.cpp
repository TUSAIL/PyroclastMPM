// BSD 3-Clause License
// Copyright (c) 2023, Retief Lubbe
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
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

#include "pyroclastmpm/boundaryconditions/rigidbodylevelset.h"

#include "rigidbodylevelset_inline.h"

namespace pyroclastmpm {

extern const int global_step_cpu;
extern const Real dt_cpu;

/// @brief Construct a new Rigid Body Level Set object
/// @param _COM center of mass of rigid body
/// @param _frames animation frames
/// @param _locations animation locations
/// @param _rotations animation rotations
RigidBodyLevelSet::RigidBodyLevelSet(const Vectorr _COM,
                                     const cpu_array<int> &_frames,
                                     const cpu_array<Vectorr> &_locations,
                                     const cpu_array<Vectorr> &_rotations)
    : num_frames((int)_frames.size()), frames_cpu(_frames),
      locations_cpu(_locations), rotations_cpu(_rotations) {

  if (DIM != 3) {
    printf("Rigid body level set only supports 3D simulations\n");
    exit(1);
  }

  if (_locations.empty()) {
    COM = _COM;
    return;
  }
  printf("Rigid body level has motion \n"); // TODO: debug

  COM = _locations[0];

  locations_cpu = _locations;

  current_frame = 0;

  euler_angles = _rotations[0];
}

/// @brief apply rigid body contact on background grid
/// @param nodes_ref Nodes container
/// @param particles_ref Particles container
void RigidBodyLevelSet::apply_on_nodes_moments(
    NodesContainer &nodes_ref, ParticlesContainer &particles_ref) {

  if (global_step_cpu == 0) {
    initialize(nodes_ref, particles_ref);
  }

  set_default_device<bool>(nodes_ref.grid.num_cells_total, {},
                           is_overlapping_gpu, false);
  // Set velocity
  // printf("frames %d \n", current_frame);
  set_velocities(particles_ref);

  // 4. Mask grid nodes that contribute to rigid body contact
  calculate_overlapping_rigidbody(nodes_ref, particles_ref);

  // 5. Get rigid body grid normals
  calculate_grid_normals(nodes_ref, particles_ref);

  // update rigid body position
  set_position(particles_ref);

  if (global_step_cpu % 400 == 0) {

    output_vtk(nodes_ref, particles_ref); // TODO: remove this , for
    //   debugging
  }

  current_frame += 1;
};

/// @brief Set the output formats
/// @param _output_formats output formats
void RigidBodyLevelSet::set_output_formats(
    const std::vector<std::string> &_output_formats) {
  output_formats = _output_formats;
}

/// @brief set velocities of rigid particles
/// @param particles_ref Particles container
void RigidBodyLevelSet::set_velocities(ParticlesContainer &particles_ref) {
  if (current_frame >= num_frames - 1) {
    return;
  }

  translational_velocity = (locations_cpu[current_frame + 1] - COM) / dt_cpu;
  // radians to degrees
  angular_velocities =
      (rotations_cpu[current_frame + 1] * (PI / 180) - euler_angles) / dt_cpu;

#ifdef CUDA_ENABLED

  KERNEL_UPDATE_RIGID_VELOCITY<<<particles_ref.launch_config.tpb,
                                 particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.velocities_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.is_rigid_gpu.data()),
      translational_velocity, COM, angular_velocities, euler_angles,
      particles_ref.num_particles);
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    update_rigid_velocity(
        particles_ref.velocities_gpu.data(), particles_ref.positions_gpu.data(),
        particles_ref.is_rigid_gpu.data(), translational_velocity, COM,
        angular_velocities, euler_angles, pid);
  }
#endif
};

/// @brief set position of rigid particles
/// @param particles_ref Particles container
void RigidBodyLevelSet::set_position(ParticlesContainer &particles_ref) {
#ifdef CUDA_ENABLED
  KERNEL_UPDATE_RIGID_POSITION<<<particles_ref.launch_config.tpb,
                                 particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocities_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.is_rigid_gpu.data()),
      particles_ref.num_particles

  );
#else

  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    update_rigid_position(particles_ref.positions_gpu.data(),
                          particles_ref.velocities_gpu.data(),
                          particles_ref.is_rigid_gpu.data(), pid);
  }
#endif

  COM += translational_velocity * dt_cpu;
  euler_angles += angular_velocities * dt_cpu;
};

/// @brief allocates memory for rigid body level set
/// @param nodes_ref Nodes container
/// @param particles_ref Particles container
void RigidBodyLevelSet::initialize(
    const NodesContainer &nodes_ref,
    [[maybe_unused]] const ParticlesContainer &particles_ref) {

  set_default_device<Vectorr>(nodes_ref.grid.num_cells_total, {}, normals_gpu,
                              Vectorr::Zero());

  set_default_device<bool>(nodes_ref.grid.num_cells_total, {},
                           is_overlapping_gpu, false);
  set_default_device<int>(nodes_ref.grid.num_cells_total, {},
                          closest_rigid_particle_gpu, -1);
}

/// @brief calculates grid normals of rigid body level set
/// @param nodes_ref Nodes container
/// @param particles_ref Particles container
void RigidBodyLevelSet::calculate_grid_normals(
    NodesContainer &nodes_ref, ParticlesContainer &particles_ref) {

#ifdef CUDA_ENABLED
  KERNELS_GRID_NORMALS_AND_NN_RIGID<<<nodes_ref.launch_config.tpb,
                                      nodes_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(normals_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.moments_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.moments_nt_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.masses_gpu.data()),
      thrust::raw_pointer_cast(is_overlapping_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.velocities_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.dpsi_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.masses_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.spatial.cell_start_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.spatial.cell_end_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.spatial.sorted_index_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.is_rigid_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
      nodes_ref.grid);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int nid = 0; nid < nodes_ref.grid.num_cells_total; nid++) {
    // TODO: fix this for CPU and 2D
    // calculate_grid_normals_nn_rigid(
    //     nodes_ref.moments_gpu.data(), nodes_ref.moments_nt_gpu.data(),
    //     nodes_ref.node_ids_gpu.data(), nodes_ref.masses_gpu.data(),
    //     is_overlapping_gpu.data(), particles_ref.velocities_gpu.data(),
    //     particles_ref.dpsi_gpu.data(), particles_ref.masses_gpu.data(),
    //     particles_ref.spatial.cell_start_gpu.data(),
    //     particles_ref.spatial.cell_end_gpu.data(),
    //     particles_ref.spatial.sorted_index_gpu.data(),
    //     particles_ref.is_rigid_gpu.data(),
    //     particles_ref.positions_gpu.data(), nodes_ref.grid, nid);
  }
#endif
}

/// @brief mask grid nodes that contribute to rigid body contact
/// @param nodes_ref Nodes container
/// @param particles_ref Particles container
void RigidBodyLevelSet::calculate_overlapping_rigidbody(
    NodesContainer &nodes_ref, ParticlesContainer &particles_ref) {
#ifdef CUDA_ENABLED
  // first pass, flag rigid body nodes
  KERNEL_GET_OVERLAPPING_RIGID_BODY_GRID<<<particles_ref.launch_config.tpb,
                                           particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(is_overlapping_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.spatial.bins_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.spatial.cell_start_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.spatial.cell_end_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.spatial.sorted_index_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.is_rigid_gpu.data()),
      nodes_ref.grid, particles_ref.num_particles);

  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    // TODO: fix this for CPU and 2D
    // get_overlapping_rigid_body_grid(
    //     is_overlapping_gpu.data(), nodes_ref.node_ids_gpu.data(),
    //     particles_ref.positions_gpu.data(),
    //     particles_ref.spatial.bins_gpu.data(),
    //     particles_ref.is_rigid_gpu.data(), nodes_ref.grid, pid);
  }
#endif
}

void RigidBodyLevelSet::output_vtk(NodesContainer &nodes_ref,
                                   ParticlesContainer &particles_ref) {
  // override in derived classes
  // printf("output vtk\n");
#if DIM > 2 // TODO: FIX THIS
  // nodes
  vtkSmartPointer<vtkPolyData> nodes_polydata =
      vtkSmartPointer<vtkPolyData>::New();

  cpu_array<Vectorr> nodes_positions_cpu = nodes_ref.give_node_coords();
  cpu_array<Vectorr> nodes_moments_cpu = nodes_ref.moments_gpu;
  cpu_array<Vectorr> nodes_moments_nt_cpu = nodes_ref.moments_nt_gpu;
  cpu_array<Vectorr> nodes_forces_external_cpu = nodes_ref.forces_external_gpu;
  cpu_array<Vectorr> nodes_forces_internal_cpu = nodes_ref.forces_internal_gpu;
  cpu_array<Vectorr> nodes_forces_total_cpu = nodes_ref.forces_total_gpu;

  cpu_array<bool> nodes_is_overlapping_cpu = is_overlapping_gpu;

  cpu_array<Vectorr> nodes_normals_cpu = normals_gpu;

  cpu_array<Real> nodes_masses_cpu = nodes_ref.masses_gpu;

  std::string format = "vtk";

  // TODO: fix this (remove and make so we can output int3)
  cpu_array<Vectori> nodes_node_ids_cpu = nodes_ref.node_ids_gpu;
  cpu_array<Vectorr> nodes_node_ids_real_cpu;
  for (int nid = 0; nid < nodes_ref.grid.num_cells_total; nid++) {
    nodes_node_ids_real_cpu.push_back(Vectorr(
        {(Real)nodes_node_ids_cpu[nid][0], (Real)nodes_node_ids_cpu[nid][1],
         (Real)nodes_node_ids_cpu[nid][2]}));
  }

  set_vtk_points(nodes_positions_cpu, nodes_polydata);
  set_vtk_pointdata<Vectorr>(nodes_normals_cpu, nodes_polydata, "Normals");
  set_vtk_pointdata<Vectorr>(nodes_node_ids_real_cpu, nodes_polydata,
                             "Nodes_id");
  set_vtk_pointdata<Vectorr>(nodes_moments_cpu, nodes_polydata, "Moments");
  set_vtk_pointdata<Vectorr>(nodes_moments_nt_cpu, nodes_polydata, "MomentsNT");
  set_vtk_pointdata<Vectorr>(nodes_forces_external_cpu, nodes_polydata,
                             "ForcesExternal");
  set_vtk_pointdata<Vectorr>(nodes_forces_internal_cpu, nodes_polydata,
                             "ForcesInternal");
  set_vtk_pointdata<Vectorr>(nodes_forces_total_cpu, nodes_polydata,
                             "ForcesTotal");
  set_vtk_pointdata<Real>(nodes_masses_cpu, nodes_polydata, "Mass");
  set_vtk_pointdata<bool>(nodes_is_overlapping_cpu, nodes_polydata,
                          "is_overlapping");
  write_vtk_polydata(nodes_polydata, "nodes_rigid_body", format);

  // particles

  vtkSmartPointer<vtkPolyData> particles_polydata =
      vtkSmartPointer<vtkPolyData>::New();

  cpu_array<Vectorr> rigid_positions_cpu = {};

  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    if (particles_ref.is_rigid_gpu[pid]) {
      rigid_positions_cpu.push_back(particles_ref.positions_gpu[pid]);
    }
  }

  int num_rigid = rigid_positions_cpu.size();
  set_vtk_points(rigid_positions_cpu, particles_polydata);
  write_vtk_polydata(particles_polydata, "rigid_particles", format);

#endif // endif DDIM=2 TODO: Remove this (or fix for 2D)
};
} // namespace pyroclastmpm