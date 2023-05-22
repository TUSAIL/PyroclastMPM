
#pragma once

#include "pyroclastmpm/boundaryconditions/rigidbodylevelset.cuh"
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/particles/particles.cuh"

// Functions to test
// [ ] RigidParticles::RigidParticles 
// [ ] RigidParticles::initialize
// [ ] RigidParticles::partition
// [ ] RigidParticles::calculate_non_rigid_grid_normals
// [ ] RigidParticles::calculate_overlapping_rigidbody
// [ ] RigidParticles::update_grid_moments
// [ ] RigidParticles::find_nearest_rigid_body
// [ ] RigidParticles::update_rigid_body
// [ ] RigidParticles::calculate_velocities
// [ ] RigidParticles::find_nearest_rigid_body
// [ ] RigidParticles::output_vtk

using namespace pyroclastmpm;

// Node that this function might change so no solid tests are written
