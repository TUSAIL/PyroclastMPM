from __future__ import annotations
from typing import Type, List, Callable

from .boundaryconditions import BoundaryCondition
from .nodes import NodesContainer
from .particles import ParticlesContainer

from .pyroclastmpm_pybind import USL as PyroUSL
from .pyroclastmpm_pybind import TLMPM as PyroTLMPM

from .pyroclastmpm_pybind import MUSL as PyroMUSL

# from .pyroclastmpm_pybind import APIC as PyroAPIC

# TODO UPDATE PYTHON DOCSTRINGS


class USL(PyroUSL):
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
            boundaryconditions=[wally0, wallx0, wallx1], # list of types BoundaryCondition
        )

    :param particles: Particles container.
    :param nodes: Nodes container
    :param boundaryconditions: A list of boundary conditions, defaults to []
    :param total_steps: Total number of simulation steps, defaults to 0
    :param output_steps: Number of steps per output, defaults to 0
    :param output_start: Step when output starts, defaults to 0
    """

    #: The nodes container :class:`pyroclastpy.nodes.NodesContainer` (interfaced with pybind11)
    nodes: Type["NodesContainer"]

    #: The particles container :class:`pyroclastpy.particles.ParticlesContainer` (interfaced with pybind11)
    particles: Type["ParticlesContainer"]

    #: List of boundaryconditions see :class:`pyroclastpy.boundaryconditions.BoundaryCondition` (interfaced with pybind11)
    boundaryconditions: List[Type["BoundaryCondition"]]

    #: The current step of the solver (interfaced with pybind11)
    current_step: int

    #: Current simulation current_time (interfaced with pybind11)
    current_time: float

    #: Total number of steps to finish (interfaced with pybind11)
    total_steps: int

    #: Step when the output start (interfaced with pybind11)
    output_start: int

    #: The nth steps to output (interfaced with pybind11)
    output_steps: int

    #: List of particles variables to be stored in a dataframe accessed with the ``state_particles`` property
    record_particles_vars: List[str] = []

    #: List of nodes variables to be stored in a dataframe accessed with the ``state_nodes`` property
    record_nodes_vars: List[str] = []

    #: A Pandas Dataframe containing the state of the particles
    state_particles = None

    #: A Pandas Dataframe containing the state of the nodes
    state_nodes = None

    def __init__(
        self,
        particles: Type["ParticlesContainer"],
        nodes: Type["NodesContainer"],
        boundaryconditions: List[Type["BoundaryCondition"]] = [],
        materials=[],
        alpha=0.99,
        total_steps: int = 0,
        output_steps: int = 0,
        output_start: int = 0,
    ):

        self.total_steps = total_steps
        self.output_steps = output_steps
        self.output_start = output_start

        #: init c++ class from pybind11
        super(USL, self).__init__(
            particles=particles,
            nodes=nodes,
            materials=materials,
            boundaryconditions=boundaryconditions,
            alpha=alpha
        )

    def solve_nsteps(self, num_steps: int):
        """Solve the simulation for a specified number of steps.

        :param num_steps: The amount of steps to run the simulation.
        """
        # calls c++ function from pybind11
        super(USL, self).solve_nsteps(num_steps)

    def run(self, callback: Callable = None):
        """Run the simulation for the specified number of steps and output steps given
        at class initiation.

        :param callback: A function to call after a loop, defaults to None
        """
        self.solve_nsteps(self.output_start)

        stop = self.current_step

        while stop < self.total_steps:
            self.solve_nsteps(self.output_steps)
            if callback is not None:
                callback()
            stop = self.current_step
            print(f"output: {stop}")


class TLMPM(PyroTLMPM):
    """
    Total Lagrangian MPM Solver

    Example usage:

    .. highlight:: python
    .. code-block:: python

        MPM = Solver(
            particles=particles, # of type ParticlesContainer
            nodes=nodes, # of type NodesContainer
            total_steps=40000,
            output_steps=1000,
            output_start=0,
            boundaryconditions=[wally0, wallx0, wallx1], # list of types BoundaryCondition
        )

    :param particles: Particles container.
    :param nodes: Nodes container
    :param boundaryconditions: A list of boundary conditions, defaults to []
    :param total_steps: Total number of simulation steps, defaults to 0
    :param output_steps: Number of steps per output, defaults to 0
    :param output_start: Step when output starts, defaults to 0
    """

    #: The nodes container :class:`pyroclastpy.nodes.NodesContainer` (interfaced with pybind11)
    nodes: Type["NodesContainer"]

    #: The particles container :class:`pyroclastpy.particles.ParticlesContainer` (interfaced with pybind11)
    particles: Type["ParticlesContainer"]

    #: List of boundaryconditions see :class:`pyroclastpy.boundaryconditions.BoundaryCondition` (interfaced with pybind11)
    boundaryconditions: List[Type["BoundaryCondition"]]

    #: The current step of the solver (interfaced with pybind11)
    current_step: int

    #: Current simulation current_time (interfaced with pybind11)
    current_time: float

    #: Total number of steps to finish (interfaced with pybind11)
    total_steps: int

    #: Step when the output start (interfaced with pybind11)
    output_start: int

    #: The nth steps to output (interfaced with pybind11)
    output_steps: int

    #: List of particles variables to be stored in a dataframe accessed with the ``state_particles`` property
    record_particles_vars: List[str] = []

    #: List of nodes variables to be stored in a dataframe accessed with the ``state_nodes`` property
    record_nodes_vars: List[str] = []

    #: A Pandas Dataframe containing the state of the particles
    state_particles = None

    #: A Pandas Dataframe containing the state of the nodes
    state_nodes = None

    def __init__(
        self,
        particles: Type["ParticlesContainer"],
        nodes: Type["NodesContainer"],
        boundaryconditions: List[Type["BoundaryCondition"]] = [],
        materials=[],
        alpha=0.99,
        total_steps: int = 0,
        output_steps: int = 0,
        output_start: int = 0,
    ):

        self.total_steps = total_steps
        self.output_steps = output_steps
        self.output_start = output_start

        #: init c++ class from pybind11
        super(TLMPM, self).__init__(
            particles=particles,
            nodes=nodes,
            materials=materials,
            boundaryconditions=boundaryconditions,
            alpha=alpha
        )

    def solve_nsteps(self, num_steps: int):
        """Solve the simulation for a specified number of steps.

        :param num_steps: The amount of steps to run the simulation.
        """
        # calls c++ function from pybind11
        super(TLMPM, self).solve_nsteps(num_steps)

    def run(self, callback: Callable = None):
        """Run the simulation for the specified number of steps and output steps given
        at class initiation.

        :param callback: A function to call after a loop, defaults to None
        """
        self.solve_nsteps(self.output_start)

        stop = self.current_step

        while stop < self.total_steps:
            self.solve_nsteps(self.output_steps)
            if callback is not None:
                callback()
            stop = self.current_step
            print(f"output: {stop}")


class MUSL(PyroMUSL):
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
            boundaryconditions=[wally0, wallx0, wallx1], # list of types BoundaryCondition
        )

    :param particles: Particles container.
    :param nodes: Nodes container
    :param boundaryconditions: A list of boundary conditions, defaults to []
    :param total_steps: Total number of simulation steps, defaults to 0
    :param output_steps: Number of steps per output, defaults to 0
    :param output_start: Step when output starts, defaults to 0
    """

    #: The nodes container :class:`pyroclastpy.nodes.NodesContainer` (interfaced with pybind11)
    nodes: Type["NodesContainer"]

    #: The particles container :class:`pyroclastpy.particles.ParticlesContainer` (interfaced with pybind11)
    particles: Type["ParticlesContainer"]

    #: List of boundaryconditions see :class:`pyroclastpy.boundaryconditions.BoundaryCondition` (interfaced with pybind11)
    boundaryconditions: List[Type["BoundaryCondition"]]

    #: The current step of the solver (interfaced with pybind11)
    current_step: int

    #: Current simulation current_time (interfaced with pybind11)
    current_time: float

    #: Total number of steps to finish (interfaced with pybind11)
    total_steps: int

    #: Step when the output start (interfaced with pybind11)
    output_start: int

    #: The nth steps to output (interfaced with pybind11)
    output_steps: int

    #: List of particles variables to be stored in a dataframe accessed with the ``state_particles`` property
    record_particles_vars: List[str] = []

    #: List of nodes variables to be stored in a dataframe accessed with the ``state_nodes`` property
    record_nodes_vars: List[str] = []

    #: A Pandas Dataframe containing the state of the particles
    state_particles = None

    #: A Pandas Dataframe containing the state of the nodes
    state_nodes = None

    def __init__(
        self,
        particles: Type["ParticlesContainer"],
        nodes: Type["NodesContainer"],
        boundaryconditions: List[Type["BoundaryCondition"]] = [],
        materials=[],
        alpha=0.99,
        total_steps: int = 0,
        output_steps: int = 0,
        output_start: int = 0,
    ):

        self.total_steps = total_steps
        self.output_steps = output_steps
        self.output_start = output_start

        #: init c++ class from pybind11
        super(MUSL, self).__init__(
            particles=particles,
            nodes=nodes,
            materials=materials,
            boundaryconditions=boundaryconditions,
            alpha=alpha
        )

    def solve_nsteps(self, num_steps: int):
        """Solve the simulation for a specified number of steps.

        :param num_steps: The amount of steps to run the simulation.
        """
        # calls c++ function from pybind11
        super(MUSL, self).solve_nsteps(num_steps)

    def run(self, callback: Callable = None):
        """Run the simulation for the specified number of steps and output steps given
        at class initiation.

        :param callback: A function to call after a loop, defaults to None
        """
        self.solve_nsteps(self.output_start)

        stop = self.current_step

        while stop < self.total_steps:
            self.solve_nsteps(self.output_steps)
            if callback is not None:
                callback()
            stop = self.current_step
            print(f"output: {stop}")
