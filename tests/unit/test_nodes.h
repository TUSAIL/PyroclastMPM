
#include "pyroclastmpm/nodes/nodes.h"

// Functions tested
// [x] NodesContainer::NodesContainer
// [x] NodesContainer::NodeContainer (get ids)
// [x] NodesContainer::integrate
// [ ] NodesContainer::NodeContainer (get types)
// [x] NodesContainer::give_coordinates (implicitly at test_nodes.py)
// [ ] NodesContainer::output_vtk

using namespace pyroclastmpm;


/**
 * @brief Construct a new TEST object for NodesContainer to test the constructor
 *
 */
TEST(NodesContainer, CONSTRUCTOR)
{

  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real nodal_spacing = 0.5;

  NodesContainer nodes = NodesContainer(min, max, nodal_spacing);

  cpu_array<Vectori> node_ids = nodes.node_ids_gpu;

#if DIM == 3
  EXPECT_EQ(nodes.num_nodes_total, 27);
  EXPECT_EQ(nodes.num_nodes[0], 3);
  EXPECT_EQ(nodes.num_nodes[1], 3);
  EXPECT_EQ(nodes.num_nodes[2], 3);
  EXPECT_EQ(node_ids[0], Vectori({0, 0, 0}));
  EXPECT_EQ(node_ids[1], Vectori({1, 0, 0}));
  EXPECT_EQ(node_ids[2], Vectori({2, 0, 0}));
  EXPECT_EQ(node_ids[3], Vectori({0, 1, 0}));
  EXPECT_EQ(node_ids[4], Vectori({1, 1, 0}));
  EXPECT_EQ(node_ids[5], Vectori({2, 1, 0}));
  EXPECT_EQ(node_ids[6], Vectori({0, 2, 0}));
  EXPECT_EQ(node_ids[7], Vectori({1, 2, 0}));
  EXPECT_EQ(node_ids[8], Vectori({2, 2, 0}));

  EXPECT_EQ(node_ids[9], Vectori({0, 0, 1}));
  EXPECT_EQ(node_ids[10], Vectori({1, 0, 1}));
  EXPECT_EQ(node_ids[11], Vectori({2, 0, 1}));
  EXPECT_EQ(node_ids[12], Vectori({0, 1, 1}));
  EXPECT_EQ(node_ids[13], Vectori({1, 1, 1}));
  EXPECT_EQ(node_ids[14], Vectori({2, 1, 1}));
  EXPECT_EQ(node_ids[15], Vectori({0, 2, 1}));
  EXPECT_EQ(node_ids[16], Vectori({1, 2, 1}));
  EXPECT_EQ(node_ids[17], Vectori({2, 2, 1}));

  EXPECT_EQ(node_ids[18], Vectori({0, 0, 2}));
  EXPECT_EQ(node_ids[19], Vectori({1, 0, 2}));
  EXPECT_EQ(node_ids[20], Vectori({2, 0, 2}));
  EXPECT_EQ(node_ids[21], Vectori({0, 1, 2}));
  EXPECT_EQ(node_ids[22], Vectori({1, 1, 2}));
  EXPECT_EQ(node_ids[23], Vectori({2, 1, 2}));
  EXPECT_EQ(node_ids[24], Vectori({0, 2, 2}));
  EXPECT_EQ(node_ids[25], Vectori({1, 2, 2}));
  EXPECT_EQ(node_ids[26], Vectori({2, 2, 2}));

#elif DIM == 2

  EXPECT_EQ(nodes.num_nodes_total, 9);
  EXPECT_EQ(nodes.num_nodes[0], 3);
  EXPECT_EQ(nodes.num_nodes[1], 3);
  EXPECT_EQ(node_ids[0], Vectori({0, 0}));
  EXPECT_EQ(node_ids[1], Vectori({1, 0}));
  EXPECT_EQ(node_ids[2], Vectori({2, 0}));
  EXPECT_EQ(node_ids[3], Vectori({0, 1}));
  EXPECT_EQ(node_ids[4], Vectori({1, 1}));
  EXPECT_EQ(node_ids[5], Vectori({2, 1}));
  EXPECT_EQ(node_ids[6], Vectori({0, 2}));
  EXPECT_EQ(node_ids[7], Vectori({1, 2}));
  EXPECT_EQ(node_ids[8], Vectori({2, 2}));

#else // DIM == 1
  EXPECT_EQ(nodes.num_nodes_total, 3);
  EXPECT_EQ(nodes.num_nodes[0], 3);
  EXPECT_EQ(node_ids[0], Vectori(0));
  EXPECT_EQ(node_ids[1], Vectori(1));
  EXPECT_EQ(node_ids[2], Vectori(2));
#endif

//   // TODO insert test for node types
}

// /**
//  * @brief Construct a new TEST object for integrating the nodes
//  *
//  */
TEST(NodesContainer, INTEGRATE)
{
  set_global_dt(0.1);
  Vectorr min = Vectorr::Zero();
  Vectorr max = Vectorr::Ones();
  Real nodal_spacing = 0.5;

  NodesContainer nodes = NodesContainer(min, max, nodal_spacing);

  cpu_array<Real> masses = nodes.masses_gpu;
  masses[0] = 1.;
  masses[1] = 2.;
  masses[2] = 3.;
  nodes.masses_gpu = masses;

  cpu_array<Vectorr> moments = nodes.moments_gpu;
  moments[0][0] = 1.;
  moments[1][0] = 1.;
  moments[2][0] = 1.;
  nodes.moments_gpu = moments;

  cpu_array<Vectorr> forces_internal =
      nodes.forces_internal_gpu;
  forces_internal[0][0] = 1.;
  forces_internal[1][0] = 2.;
  forces_internal[2][0] = 3.;
  nodes.forces_internal_gpu = forces_internal;

  cpu_array<Vectorr> forces_external =
      nodes.forces_external_gpu;
  forces_external[0][0] = 1.;
  forces_external[1][0] = 2.;
  forces_external[2][0] = 3.;
  nodes.forces_external_gpu = forces_external;

  nodes.integrate();

  cpu_array<Vectorr> forces_total = nodes.forces_total_gpu;
  EXPECT_NEAR(forces_total[0][0], 2., 0.000001);
  EXPECT_NEAR(forces_total[1][0], 4., 0.000001);
  EXPECT_NEAR(forces_total[2][0], 6., 0.000001);

  cpu_array<Vectorr> moments_nt = nodes.moments_nt_gpu;
  EXPECT_NEAR(moments_nt[0][0], 1.2, 0.000001);
  EXPECT_NEAR(moments_nt[1][0], 1.4, 0.000001);
  EXPECT_NEAR(moments_nt[2][0], 1.6, 0.000001);

}