#pragma once

__device__ __host__ inline Real cubic(const Real relative_distance, const int node_type = 0)
{
  // 0 and N
  if (node_type == 1)
  {

    if (relative_distance >= -2.0 && relative_distance < -1.0)
    {
      return ((1.0 / 6.0 * relative_distance + 1.0) * relative_distance + 2.0) * relative_distance + 4.0 / 3.0;
    }
    else if (relative_distance >= -1 && relative_distance < 0.)
    {
      return (-1.0 / 6.0 * relative_distance * relative_distance + 1.0) * relative_distance + 1.0;
    }
    else if (relative_distance >= 0. && relative_distance < 1.0)
    {
      return (1.0 / 6.0 * relative_distance * relative_distance - 1.0) * relative_distance + 1.0;
    }
    else if (relative_distance >= 1.0 && relative_distance < 2.0)
    {

      return ((-1.0 / 6.0 * relative_distance + 1.) * relative_distance - 2.) * relative_distance + 4.0 / 3.0;
    }
    return 0.0;
  }
  else if (node_type == 2) // 0 + h
  {
    // right side of boundary
    if (relative_distance >= -1.0 && relative_distance < 0.0)
    {
      return (-1.0 / 3.0 * relative_distance - 1.) * relative_distance * relative_distance + 2.0 / 3.0;
    }
    else if (relative_distance >= 0. && relative_distance < 1.)
    {
      return (0.5 * relative_distance - 1.) * relative_distance * relative_distance + 2.0 / 3.0;
    }
    else if (relative_distance >= 1. && relative_distance < 2.)
    {

      return ((-1.0 / 6.0 * relative_distance + 1) * relative_distance - 2) * relative_distance + 4.0 / 3.0;
    }
    return 0.0;
  }
  else if (node_type == 3) // cells in the middle
  {
    if (relative_distance >= -2. && relative_distance < -1.)
    {
      return ((1.0 / 6.0 * relative_distance + 1.) * relative_distance + 2.) * relative_distance + 4.0 / 3.0;
    }
    else if (relative_distance >= -1 && relative_distance < 0)
    {
      return (-0.5 * relative_distance - 1) * relative_distance * relative_distance + 2.0 / 3.0;
    }
    else if (relative_distance >= 0. && relative_distance < 1.)
    {
      return (0.5 * relative_distance - 1) * relative_distance * relative_distance + 2.0 / 3.0;
    }
    else if (relative_distance >= 1. && relative_distance < 2.)
    {
      return ((-1.0 / 6.0 * relative_distance + 1.) * relative_distance - 2.) * relative_distance + 4.0 / 3.0;
    }

    return 0.0;
  }
  else if (node_type == 4)
  {

    if (relative_distance >= -2. && relative_distance < -1.)
    {
      return ((1.0 / 6.0 * relative_distance + 1) * relative_distance + 2) * relative_distance + 4.0 / 3.0;
    }
    else if (relative_distance >= -1. && relative_distance < 0.)
    {
      return (-0.5 * relative_distance - 1) * relative_distance * relative_distance + 2.0 / 3.0;
    }
    else if (relative_distance >= 0. && relative_distance < 1.)
    {
      return (1.0 / 3.0 * relative_distance - 1.) * relative_distance * relative_distance + 2.0 / 3.0;
    }

    return 0.0;
  }
  return 0.0;
}

__device__ __host__ inline Real derivative_cubic(const Real relative_distance, const Real inv_cellsize, const int node_type = 0)
{
  if (node_type == 1)
  {

    if (relative_distance >= -2.0 && relative_distance < -1.0)
    {
      return inv_cellsize * ((0.5 * relative_distance + 2.) * relative_distance + 2.);
    }
    else if (relative_distance >= -1 && relative_distance < 0.)
    {
      return inv_cellsize * (-0.5 * relative_distance * relative_distance + 1.);
    }
    else if (relative_distance >= 0. && relative_distance < 1.0)
    {
      return inv_cellsize * (0.5 * relative_distance * relative_distance - 1.);
    }
    else if (relative_distance >= 1.0 && relative_distance < 2.0)
    {
      return inv_cellsize * ((-0.5 * relative_distance + 2) * relative_distance - 2.);
    }
    return 0.0;
  }
  else if (node_type == 2) // 0 + h
  {

    // right side of boundary
    if (relative_distance >= -1.0 && relative_distance < 0.0)
    {

      return inv_cellsize * (-relative_distance - 2) * relative_distance;
    }
    else if (relative_distance >= 0. && relative_distance < 1.)
    {

      return inv_cellsize * (3.0 / 2.0 * relative_distance - 2.) * relative_distance;
    }
    else if (relative_distance >= 1. && relative_distance < 2.)
    {
      return inv_cellsize * ((-0.5 * relative_distance + 2) * relative_distance - 2.);
    }
    return 0.0;
  }
  else if (node_type == 3) // cells in the middle
  {
    if (relative_distance >= -2. && relative_distance < -1.)
    {
      return inv_cellsize * ((0.5 * relative_distance + 2.) * relative_distance + 2.);
    }
    else if (relative_distance >= -1 && relative_distance < 0)
    {
      return inv_cellsize * (-3.0 / 2.0 * relative_distance - 2.) * relative_distance;
    }
    else if (relative_distance >= 0. && relative_distance < 1.)
    {
      return inv_cellsize * (3.0 / 2.0 * relative_distance - 2.) * relative_distance;
    }
    else if (relative_distance >= 1. && relative_distance < 2.)
    {
      return inv_cellsize * ((-0.5 * relative_distance + 2) * relative_distance - 2.);
    }

    return 0.0;
  }
  else if (node_type == 4)
  {

    if (relative_distance >= -2.0 && relative_distance < -1.0)
    {
      return inv_cellsize * ((0.5 * relative_distance + 2.) * relative_distance + 2.);
    }
    else if (relative_distance >= -1 && relative_distance < 0)
    {
      return inv_cellsize * (-3.0 / 2.0 * relative_distance - 2.) * relative_distance;
    }
    else if (relative_distance >= 0. && relative_distance < 1.)
    {
      return inv_cellsize * relative_distance * (relative_distance - 2);
    }

    return 0.0;
  }

  return 0.0;
}
