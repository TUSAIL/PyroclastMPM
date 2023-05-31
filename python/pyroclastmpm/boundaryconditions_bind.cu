#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

#include "pyroclastmpm/common/types_common.cuh"
#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
#include "pyroclastmpm/boundaryconditions/gravity.cuh"
#include "pyroclastmpm/boundaryconditions/bodyforce.cuh"
#include "pyroclastmpm/boundaryconditions/rigidbodylevelset.cuh"
#include "pyroclastmpm/boundaryconditions/planardomain.cuh"
#include "pyroclastmpm/boundaryconditions/nodedomain.cuh"

namespace py = pybind11;

namespace pyroclastmpm
{

    void boundaryconditions_module(py::module &m)
    {
        py::class_<BoundaryCondition>(m, "BoundaryCondition").def(py::init<>());

        py::class_<BodyForce>(m, "BodyForce")
            .def(py::init<std::string, std::vector<Vectorr>, std::vector<bool>>(),
                 py::arg("mode"), py::arg("values"), py::arg("mask"))
            .def_readwrite("mode_id", &BodyForce::mode_id);

        py::class_<Gravity>(m, "Gravity")
            .def(py::init<Vectorr, bool, int, Vectorr>(), py::arg("gravity"),
                 py::arg("is_ramp"), py::arg("ramp_step"), py::arg("gravity_end"))
            .def_readwrite("gravity", &Gravity::gravity);

        py::class_<RigidBodyLevelSet>(m, "RigidBodyLevelSet")
            .def(py::init<Vectorr,
                          std::vector<int>,
                          std::vector<Vectorr>, std::vector<Vectorr>, std::vector<OutputType>>(),
                 py::arg("COM") = Vectorr::Zero(),
                 py::arg("frames") = std::vector<int>(),
                 py::arg("locations") = std::vector<Vectorr>(),
                 py::arg("rotations") = std::vector<Vectorr>(),
                 py::arg("output_formats") = std::vector<OutputType>());

        py::class_<PlanarDomain>(m, "PlanarDomain")
            .def(py::init<Vectorr, Vectorr>(),
                 py::arg("axis0_friction") = Vectorr::Zero(), py::arg("axis1_friction") = Vectorr::Zero());

        py::class_<NodeDomain>(m, "NodeDomain")
            .def(py::init<Vectori, Vectori>(),
                 py::arg("axis0_mode") = Vectori::Zero(), py::arg("axis1_mode") = Vectori::Zero());
    };

} // namespace pyroclastmpm