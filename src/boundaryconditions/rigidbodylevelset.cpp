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

namespace pyroclastmpm {

#ifdef CUDA_ENABLED
extern __constant__ Real dt_gpu;
extern __constant__ int num_surround_nodes_gpu;
extern __constant__ int g2p_window_gpu[64][3];
extern __constant__ int p2g_window_gpu[64][3];
#else
extern int num_surround_nodes_cpu;
extern int g2p_window_cpu[64][3];
extern int p2g_window_cpu[64][3];
#endif

extern int global_step_cpu;

#ifdef CUDA_ENABLED
extern Real __constant__ dt_gpu;
#endif
extern Real dt_cpu;

// include private header with kernels here to inline theu
#include "rigidbodylevelset_inline.h"

RigidBodyLevelSet::RigidBodyLevelSet(const Vectorr _COM,
                                     const cpu_array<int> _frames,
                                     const cpu_array<Vectorr> _locations,
                                     const cpu_array<Vectorr> _rotations)
    : frames_cpu(_frames), locations_cpu(_locations),
      rotations_cpu(_rotations) {
  num_frames = _frames.size();

  if (DIM != 3) {
    printf("Rigid body level set only supports 3D simulations\n");
    exit(1);
  }
  COM = _locations[0];
  current_frame = 0;

  euler_angles = _rotations[0];
}
void RigidBodyLevelSet::apply_on_nodes_moments(
    NodesContainer &nodes_ref, ParticlesContainer &particles_ref) {

  if (global_step_cpu == 0) {
    initialize(nodes_ref, particles_ref);
  }
  // Set velocity

  set_velocities(particles_ref);

  // 4. Mask grid nodes that do not contribute to rigid body contact
  calculate_overlapping_rigidbody(nodes_ref, particles_ref);

  // 5. Get rigid body grid normals
  calculate_grid_normals(nodes_ref, particles_ref);

  // update rigid body position
  set_position(particles_ref);

  current_frame += 1;
};

void RigidBodyLevelSet::set_output_formats(
    const std::vector<std::string> &_output_formats) {
  output_formats = _output_formats;
}

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

void RigidBodyLevelSet::initialize(NodesContainer &nodes_ref,
                                   ParticlesContainer &particles_ref) {
  set_default_device<Vectorr>(nodes_ref.num_nodes_total, {}, normals_gpu,
                              Vectorr::Zero());
  set_default_device<bool>(nodes_ref.num_nodes_total, {}, is_overlapping_gpu,
                           false);
  set_default_device<int>(nodes_ref.num_nodes_total, {},
                          closest_rigid_particle_gpu, -1);
}

void RigidBodyLevelSet::calculate_grid_normals(
    NodesContainer &nodes_ref, ParticlesContainer &particles_ref) {

#ifdef CUDA_ENABLED
  KERNELS_GRID_NORMALS_AND_NN_RIGID<<<nodes_ref.launch_config.tpb,
                                      nodes_ref.launch_config.bpg>>>(
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
      nodes_ref.node_start, nodes_ref.inv_node_spacing,
      particles_ref.spatial.num_cells, particles_ref.spatial.num_cells_total);
  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int nid = 0; nid < nodes_ref.num_nodes_total; nid++) {
    calculate_grid_normals_nn_rigid(
        nodes_ref.moments_gpu.data(), nodes_ref.moments_nt_gpu.data(),
        nodes_ref.node_ids_gpu.data(), nodes_ref.masses_gpu.data(),
        is_overlapping_gpu.data(), particles_ref.velocities_gpu.data(),
        particles_ref.dpsi_gpu.data(), particles_ref.masses_gpu.data(),
        particles_ref.spatial.cell_start_gpu.data(),
        particles_ref.spatial.cell_end_gpu.data(),
        particles_ref.spatial.sorted_index_gpu.data(),
        particles_ref.is_rigid_gpu.data(), particles_ref.positions_gpu.data(),
        nodes_ref.node_start, nodes_ref.inv_node_spacing,
        particles_ref.spatial.num_cells, particles_ref.spatial.num_cells_total,
        nid);
  }
#endif
}

void RigidBodyLevelSet::calculate_overlapping_rigidbody(
    NodesContainer &nodes_ref, ParticlesContainer &particles_ref) {
#ifdef CUDA_ENABLED
  KERNEL_GET_OVERLAPPING_RIGID_BODY_GRID<<<particles_ref.launch_config.tpb,
                                           particles_ref.launch_config.bpg>>>(
      thrust::raw_pointer_cast(is_overlapping_gpu.data()),
      thrust::raw_pointer_cast(nodes_ref.node_ids_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.positions_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.spatial.bins_gpu.data()),
      thrust::raw_pointer_cast(particles_ref.is_rigid_gpu.data()),
      particles_ref.spatial.num_cells, particles_ref.spatial.grid_start,
      particles_ref.spatial.inv_cell_size,
      particles_ref.spatial.num_cells_total, particles_ref.num_particles);

  gpuErrchk(cudaDeviceSynchronize());
#else
  for (int pid = 0; pid < particles_ref.num_particles; pid++) {
    get_overlapping_rigid_body_grid(
        is_overlapping_gpu.data(), nodes_ref.node_ids_gpu.data(),
        particles_ref.positions_gpu.data(),
        particles_ref.spatial.bins_gpu.data(),
        particles_ref.is_rigid_gpu.data(), particles_ref.spatial.num_cells,
        particles_ref.spatial.grid_start, particles_ref.spatial.inv_cell_size,
        particles_ref.spatial.num_cells_total, pid);
  }
#endif
}

} // namespace pyroclastmpm