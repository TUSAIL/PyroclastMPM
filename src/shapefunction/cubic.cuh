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