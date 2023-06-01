#pragma once

#include "pyroclastmpm/boundaryconditions/planardomain.h"
#include "pyroclastmpm/particles/particles.h"

// Functions to test
// [ ] PlanarDomain::PlanarDomain ("x0") // different axis
// [ ] PlanarDomain::apply_on_particles(touching the boundary and not touching
// the boundary) [ ] PlanarDomain::apply_on_particles (friction and not)

using namespace pyroclastmpm;

TEST(PlanarDomain, apply_on_particles) {

  Vectorr min = Vectorr::Zero();

  Vectorr max = Vectorr::Ones();

  Real cell_size = 0.1;

#if DIM == 3
  const std::vector<Vectorr> pos = {
      Vectorr({0.1, 0.1, 0.0}), Vectorr({0.199, 0.1, 0.0}),
      Vectorr({0.82, 0.0, 1.}), Vectorr({0.82, 0.6, 0.})};
#elif DIM == 2
  const std::vector<Vectorr> pos = {Vectorr({0.1, 0.1}), Vectorr({0.199, 0.1}),
                                    Vectorr({0.82, 0.0}), Vectorr({0.82, 0.6})};
#else
  const std::vector<Vectorr> pos = {Vectorr(0.1), Vectorr(0.2)};
#endif

  set_globals(0.1, 1, LinearShapeFunction, "output");

  ParticlesContainer particles = ParticlesContainer(pos);

  particles.set_spatialpartition(min, max, cell_size);

  // no friction
  PlanarDomain planardomain(Vectorr::Zero(), Vectorr::Zero());
}