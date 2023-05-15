
#pragma once

#include "pyroclastmpm/spatialpartition/spatialpartition.cuh"

using namespace pyroclastmpm;

// Functions tested
// [x] SpatialPartition::SpatialPartition
// [x] SpatialPartition::calculate_hash
// [x] SpatialPartition::sort_hash
// [x] SpatialPartition::bin_particles
// [ ] SpatialPartition::reset

/**
 * @brief Construct a new TEST object for SpatialPartitioning object constructor
 * in 3D
 *
 */
TEST(SpatialPartition, CONSTRUCTOR)
{

  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real cell_size = 0.5;
  int num_elements = 2;

  SpatialPartition spatial = SpatialPartition(min, max, cell_size, num_elements);

#if DIM == 3
  EXPECT_EQ(spatial.num_cells_total, 27);
  EXPECT_EQ(spatial.num_cells[0], 3);
  EXPECT_EQ(spatial.num_cells[1], 3);
  EXPECT_EQ(spatial.num_cells[2], 3);

#elif DIM == 2

  EXPECT_EQ(spatial.num_cells_total, 9);
  EXPECT_EQ(spatial.num_cells[0], 3);
  EXPECT_EQ(spatial.num_cells[1], 3);

#else // DIM == 1
  EXPECT_EQ(spatial.num_cells_total, 3);
  EXPECT_EQ(spatial.num_cells[0], 3);
#endif
}

/**
 * @brief Construct a new TEST object for calculating the hash of particles in
 * 3D
 *
 */
TEST(SpatialPartition, CALC_HASH)
{

  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real cell_size = 0.5;

#if DIM == 3
  std::vector<Vectorr> pos = {Vectorr({0., 0.25, 0.}), Vectorr({0.8, 0.6, 0.4}), Vectorr({0., 0.7, 0.})};
#elif DIM == 2
  std::vector<Vectorr> pos = {
      Vectorr({0., 0.25}),
      Vectorr({0.8, 0.6}), Vectorr({0., 0.7})};
#else // DIM == 1
  std::vector<Vectorr> pos = {
      Vectorr(0.), Vectorr(0.8), Vectorr(0.)};
#endif

  gpu_array<Vectorr> pos_gpu(pos);

  SpatialPartition spatial = SpatialPartition(min, max, cell_size, pos_gpu.size());

  spatial.reset();

  spatial.calculate_hash(pos_gpu);

  cpu_array<Vectori> bins = spatial.bins_gpu;
  cpu_array<unsigned int> hash_unsorted =
      spatial.hash_unsorted_gpu;

#if DIM == 3
  // test binning
  EXPECT_EQ(bins[0][0], 0);
  EXPECT_EQ(bins[0][1], 0);
  EXPECT_EQ(bins[0][2], 0);
  EXPECT_EQ(bins[1][0], 1);
  EXPECT_EQ(bins[1][1], 1);
  EXPECT_EQ(bins[1][2], 0);
  EXPECT_EQ(bins[2][0], 0);
  EXPECT_EQ(bins[2][1], 1);
  EXPECT_EQ(bins[2][2], 0);

  // test hash 3D - x + y*Nx + z*Nx*Ny
  EXPECT_EQ(hash_unsorted[0], 0);
  EXPECT_EQ(hash_unsorted[1], 4);
  EXPECT_EQ(hash_unsorted[2], 3);

#elif DIM == 2
  // test binning
  EXPECT_EQ(bins[0][0], 0);
  EXPECT_EQ(bins[0][1], 0);
  EXPECT_EQ(bins[1][0], 1);
  EXPECT_EQ(bins[1][1], 1);
  EXPECT_EQ(bins[2][0], 0);
  EXPECT_EQ(bins[2][1], 1);

  // test hash 2D- x + y*Nx
  EXPECT_EQ(hash_unsorted[0], 0);
  EXPECT_EQ(hash_unsorted[1], 4);
  EXPECT_EQ(hash_unsorted[2], 3);

#else // DIM == 1

  // test binning
  EXPECT_EQ(bins[0][0], 0);
  EXPECT_EQ(bins[1][0], 1);
  EXPECT_EQ(bins[2][0], 0);

  // test hash 1D hash = bin
  EXPECT_EQ(hash_unsorted[0], 0);
  EXPECT_EQ(hash_unsorted[1], 1);
  EXPECT_EQ(hash_unsorted[2], 0);

#endif
}

/**
 * @brief Construct a new TEST object for sorting the hashes and keys
 *
 */
TEST(SpatialPartition, SORT_HASH)
{
  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real cell_size = 0.5;

#if DIM == 3
  std::vector<Vectorr> pos = {Vectorr({0., 0.25, 0.}), Vectorr({0.8, 0.6, 0.4}), Vectorr({0., 0.7, 0.})};
#elif DIM == 2
  std::vector<Vectorr> pos = {
      Vectorr({0., 0.25}),
      Vectorr({0.8, 0.6}), Vectorr({0., 0.7})};
#else // DIM == 1
  std::vector<Vectorr> pos = {
      Vectorr(0.), Vectorr(0.8), Vectorr(0.)};
#endif

  gpu_array<Vectorr> pos_gpu(pos);

  SpatialPartition spatial = SpatialPartition(min, max, cell_size, pos_gpu.size());

  spatial.reset();

  spatial.calculate_hash(pos_gpu);

  spatial.sort_hashes();

  cpu_array<unsigned int> sorted_index =
      spatial.sorted_index_gpu;
  cpu_array<unsigned int> hash_sorted =
      spatial.hash_sorted_gpu;

#if DIM == 3
  // sorted hash
  EXPECT_EQ(hash_sorted[0], 0);
  EXPECT_EQ(hash_sorted[1], 3);
  EXPECT_EQ(hash_sorted[2], 4);
  // sorted index
  EXPECT_EQ(sorted_index[0], 0);
  EXPECT_EQ(sorted_index[1], 2);
  EXPECT_EQ(sorted_index[2], 1);

#elif DIM == 2
  // test binning
  // sorted hash
  EXPECT_EQ(hash_sorted[0], 0);
  EXPECT_EQ(hash_sorted[1], 3);
  EXPECT_EQ(hash_sorted[2], 4);
  // sorted index
  EXPECT_EQ(sorted_index[0], 0);
  EXPECT_EQ(sorted_index[1], 2);
  EXPECT_EQ(sorted_index[2], 1);
#else // DIM == 1

  // sorted hash
  EXPECT_EQ(hash_sorted[0], 0);
  EXPECT_EQ(hash_sorted[1], 0);
  EXPECT_EQ(hash_sorted[2], 1);
  // sorted index
  EXPECT_EQ(sorted_index[0], 0);
  EXPECT_EQ(sorted_index[1], 2);
  EXPECT_EQ(sorted_index[2], 1);
#endif
}

/**
 * @brief Construct a new TEST object for binning the particles
 *
 */
TEST(SpatialPartition, BIN_PARTICLES)
{
  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real cell_size = 0.5;

#if DIM == 3
  std::vector<Vectorr> pos = {Vectorr({0., 0.25, 0.}), Vectorr({0.8, 0.6, 0.4}), Vectorr({0., 0.7, 0.})};
#elif DIM == 2
  std::vector<Vectorr> pos = {
      Vectorr({0., 0.25}),
      Vectorr({0.8, 0.6}), Vectorr({0., 0.7})};
#else // DIM == 1
  std::vector<Vectorr> pos = {
      Vectorr(0.), Vectorr(0.8), Vectorr(0.)};
#endif

  gpu_array<Vectorr> pos_gpu(pos);

  SpatialPartition spatial = SpatialPartition(min, max, cell_size, pos_gpu.size());

  spatial.reset();

  spatial.calculate_hash(pos_gpu);

  spatial.sort_hashes();

  spatial.bin_particles();

  cpu_array<int> cell_start = spatial.cell_start_gpu;
  cpu_array<int> cell_end = spatial.cell_end_gpu;

#if DIM == 3
  for (int nid = 0; nid < 27; nid++)
  {
    if (nid == 0)
    {
      EXPECT_EQ(cell_start[nid], 0);
      EXPECT_EQ(cell_end[nid], 1);
    }
    else if (nid == 3)
    {
      EXPECT_EQ(cell_start[nid], 1);
      EXPECT_EQ(cell_end[nid], 2);
    }
    else if (nid == 4)
    {
      EXPECT_EQ(cell_start[nid], 2);
      EXPECT_EQ(cell_end[nid], 3);
    }
    else
    {
      EXPECT_EQ(cell_start[nid], -1);
      EXPECT_EQ(cell_end[nid], -1);
    }
  }
#elif DIM == 2
  for (int nid = 0; nid < 9; nid++)
  {
    if (nid == 0)
    {
      EXPECT_EQ(cell_start[nid], 0);
      EXPECT_EQ(cell_end[nid], 1);
    }
    else if (nid == 3)
    {
      EXPECT_EQ(cell_start[nid], 1);
      EXPECT_EQ(cell_end[nid], 2);
    }
    else if (nid == 4)
    {
      EXPECT_EQ(cell_start[nid], 2);
      EXPECT_EQ(cell_end[nid], 3);
    }
    else
    {
      EXPECT_EQ(cell_start[nid], -1);
      EXPECT_EQ(cell_end[nid], -1);
    }
  }
#else // DIM == 1
  for (int nid = 0; nid < 3; nid++)
  {
    if (nid == 0)
    {
      EXPECT_EQ(cell_start[nid], 0);
      EXPECT_EQ(cell_end[nid], 2);
    }
    else if (nid == 1)
    {
      EXPECT_EQ(cell_start[nid], 2);
      EXPECT_EQ(cell_end[nid], 3);
    }
    else if (nid == 2)
    {
      EXPECT_EQ(cell_start[nid], -1);
      EXPECT_EQ(cell_end[nid], -1);
    }
  }
#endif
}