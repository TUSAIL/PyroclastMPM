from __future__ import annotations

import typing as t

from . import pyroclastmpm_pybind as MPM
from .boundaryconditions import BoundaryCondition
from .materials import Material
from .nodes import NodesContainer
from .particles import ParticlesContainer

# trunk-ignore-all(flake8/E501)
# trunk-ignore-all(ruff/E501)


class USL(MPM.USL):
    """
    Update Stress Last Solver

    Example usage:

    .. highlight:: python
    .. code-block:: python

        MPM = Solver(
            particles=particles, # of type ParticlesContainer
            nodes=nodes, # of type NodesContainer
            total_steps=40000,
            output_steps=1000,
            output_start=0,
            boundaryconditions=[gravity], # list of types BoundaryCondition
        )
    """

    #: Total number of steps to finish (interfaced with pybind11)
    total_steps: int

    #: Step when the output start (interfaced with pybind11)
    output_start: int

    #: The nth steps to output (interfaced with pybind11)
    output_steps: int

    def __init__(
        self,
        particles: t.Type["ParticlesContainer"],
        nodes: t.Type["NodesContainer"],
        boundaryconditions: t.List[t.Type["BoundaryCondition"]] = None,
        materials: t.List[t.Type["Material"]] = None,
        alpha=0.99,
        total_steps: int = 0,
        output_steps: int = 0,
        output_start: int = 0,
    ):
        """Initialize Solver

        Args:
            particles (t.Type[&quot;ParticlesContainer&quot;]):
                                    Particles container object
            nodes (t.Type[&quot;NodesContainer&quot;]):
                                    Nodes container object
            boundaryconditions (t.List[t.Type[&quot;BoundaryCondition&quot;]], optional):
                                    List of boundary conditions.
                                    Defaults to None.
            materials (_type_, optional): List of materials. Defaults to None.
            alpha (float, optional): Ratio of FLIP/PIC. Defaults to 0.99.
            total_steps (int, optional): Total number of steps to solve.
                                         Defaults to 0.
            output_steps (int, optional): Save files for every i'th step. Defaults to 0.
            output_start (int, optional): The step at which the output starts. Defaults to 0.
        """
        if boundaryconditions is None:
            boundaryconditions = []
        if materials is None:
            materials = []

        self.total_steps = total_steps
        self.output_steps = output_steps
        self.output_start = output_start

        #: init c++ class from pybind11
        super(USL, self).__init__(
            particles=particles,
            nodes=nodes,
            materials=materials,
            boundaryconditions=boundaryconditions,
            alpha=alpha,
        )

    def solve_nsteps(self, num_steps: int):
        """Solve the simulation for a specified number of steps.

        Args:
            num_steps (int): Number of steps to solve
        """
        super(USL, self).solve_nsteps(num_steps)

    def run(self, callback: t.Callable = None):
        """Run the simulation for the specified number of steps
        Args:
            callback (t.Callable, optional): Callback function called after every i'th (output) step.
                         Defaults to None.
        """
        self.solve_nsteps(self.output_start)

        stop = self.current_step

        while stop < self.total_steps:
            self.solve_nsteps(self.output_steps)
            if callback is not None:
                callback()
            stop = self.current_step
            print(f"output: {stop}")
