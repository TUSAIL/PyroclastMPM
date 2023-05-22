// pybind
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

#include "pyroclastmpm/common/types_common.cuh"

// SOLVERS
#include "pyroclastmpm/solver/solver.cuh"
#include "pyroclastmpm/solver/usl/usl.cuh"
// #include "pyroclastmpm/solver/tlmpm/tlmpm.cuh"
// #include "pyroclastmpm/solver/musl/musl.cuh"
// #include "pyroclastmpm/solver/apic/apic.cuh"

namespace py = pybind11;

namespace pyroclastmpm
{

    extern const Real dt_cpu;

    extern const int global_step_cpu;

    void solver_module(py::module &m)
    {
        // SOLVER BASE
        py::class_<Solver>(m, "Solver")
            .def(py::init<ParticlesContainer, NodesContainer,
                          std::vector<MaterialType>,
                          std::vector<BoundaryConditionType>>(),
                 py::arg("particles"), py::arg("nodes"),
                 py::arg("materials") = std::vector<MaterialType>(),
                 py::arg("boundaryconditions") =
                     std::vector<BoundaryConditionType>()) // INIT
            .def("solve_nsteps", &Solver::solve_nsteps)
            .def_readwrite("nodes", &Solver::nodes)         // NODES
            .def_readwrite("particles", &Solver::particles) // PARTICLES
            .def_property("current_time",
                          [](Solver &self)
                          { return dt_cpu * global_step_cpu; },
                          {}) // CURRENT TIME
            .def_property("current_step",
                          [](Solver &self)
                          { return global_step_cpu; },
                          {}) // CURRENT TIME
            .def_property(
                "boundaryconditions",
                [](Solver &self)
                {
                    return std::vector<BoundaryConditionType>(
                        self.boundaryconditions.begin(), self.boundaryconditions.end());
                }, // getter
                [](Solver &self, const std::vector<BoundaryConditionType> &value)
                {
                    self.boundaryconditions = value;
                } // setter
            );    // BOUNDARY CONDTIONS

        // USL SOLVER
        py::class_<USL, Solver>(m, "USL").def(
            py::init<ParticlesContainer, NodesContainer, std::vector<MaterialType>,
                     std::vector<BoundaryConditionType>, Real>(),
            py::arg("particles"), py::arg("nodes"),
            py::arg("materials") = std::vector<MaterialType>(),
            py::arg("boundaryconditions") =
                std::vector<BoundaryConditionType>(),
            py::arg("alpha") = 0.99); // INIT

        // // MUSL SOLVER
        // py::class_<MUSL, USL>(m, "MUSL").def(
        //     py::init<ParticlesContainer, NodesContainer, std::vector<MaterialType>,
        //              std::vector<BoundaryConditionType>, Real>(),
        //     py::arg("particles"), py::arg("nodes"),
        //     py::arg("materials") = std::vector<MaterialType>(),
        //     py::arg("boundaryconditions") =
        //         std::vector<BoundaryConditionType>(),
        //     py::arg("alpha") = 0.99); // INIT

        // // TLMPM SOLVER
        // py::class_<TLMPM, MUSL>(m, "TLMPM").def(py::init<ParticlesContainer, NodesContainer, std::vector<MaterialType>, std::vector<BoundaryConditionType>, Real>(), py::arg("particles"), py::arg("nodes"), py::arg("materials") = std::vector<MaterialType>(), py::arg("boundaryconditions") = std::vector<BoundaryConditionType>(), py::arg("alpha") = 0.99); // INIT

        // // APIC SOLVER
        // py::class_<APIC, Solver>(m, "APIC").def(
        //     py::init<ParticlesContainer, NodesContainer, std::vector<MaterialType>,
        //              std::vector<BoundaryConditionType>>(),
        //     py::arg("particles"), py::arg("nodes"), py::arg("materials"),
        //     py::arg("boundaryconditions") = std::vector<BoundaryConditionType>());
    }
} // namespace pyroclastmpm
