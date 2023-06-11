#pragma once

__device__ __host__ inline Real linear(const Real relative_distance, const int node_type = 0)
{
  double abs_relative_distance = fabs(relative_distance);
  if (abs_relative_distance >= 1.0)
  {
    return 0.0;
  }

  return 1.0 - abs_relative_distance;
}

__device__ __host__ inline Real derivative_linear(const Real relative_distance, const Real inv_cellsize, const int node_type = 0)
{
  // if (relative_distance >= 1.0 || relative_distance <= -1.0 || relative_distance == 0.0)
  if (relative_distance > 1.0 || relative_distance < -1.0 || relative_distance == 0.0)
  {
    return 0.0;
  }
  else if (relative_distance > 0.0)
  {

    return -inv_cellsize;
  }

  return inv_cellsize;
}