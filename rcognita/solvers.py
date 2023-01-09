from abc import ABC, abstractmethod
import warnings

try:
    import casadi
except ModuleNotFoundError:
    pass

import numpy as np
from .systems import System
from .__utilities import rc


class Solver(ABC):
    """
    Solver is an abstract class representing a solver for optimization problems.

    Attributes:
    y: A property representing the output of the solver.
    t: A property representing the current time of the solver.

    Methods:
    step: An abstract method representing a single step of the solver.
    """

    @property
    @abstractmethod
    def y(self):
        pass

    @property
    @abstractmethod
    def t(self):
        pass

    @abstractmethod
    def step(self):
        """
        Advance the solver by one step.
        """
        pass


class CasADiSolver(Solver):
    """
    The CasADiSolver class is a subclass of the abstract Solver class that allows for the integration of a system of differential equations using the CasADi library. It can be used to solve a system of equations with a given initial state and action, and a given step size and final time. The CasADiSolver class has several properties, including the integrator object, the starting and ending times for the integration, the step size for the integration, the initial and current states of the system, and the initial and current actions applied to the system. It also has a step() method which advances the integration by one time step and updates the current state of the system.
    """

    def __init__(
        self,
        integrator,
        time_start: float,
        time_final: float,
        step_size: float,
        state_init: np.array,
        action_init: np.array,
        system: System,
    ):
        """
        Initialize a CasADiSolver object.

        :param integrator: A CasADi integrator object.
        :type integrator: casadi.integrator
        :param time_start: The starting time for the solver.
        :type time_start: float
        :param time_final: The final time for the solver.
        :type time_final: float
        :param step_size: The step size for the solver.
        :type step_size: float
        :param state_init: The initial state for the solver.
        :type state_init: np.array
        :param action_init: The initial action for the solver.
        :type action_init: np.array
        :param system: The system object for the solver.
        :type system: System
        """
        self.integrator = integrator
        self.time_start = time_start
        self.time_final = time_final
        self.step_size = step_size
        self.time = self.time_start
        self.state_init = state_init
        self.state = self.state_init
        self.state_new = self.state
        self.action_init = action_init
        self.action = self.action_init
        self.system = system

    def step(self):
        """
        Advance the solver by one step.
        """
        if self.time >= self.time_final:
            raise RuntimeError("An attempt to step with a finished solver")
        self.state_new = np.squeeze(
            self.integrator(x0=self.state, p=self.system.action)["xf"].full()
        )
        self.time += self.step_size
        self.state = self.state_new

    @property
    def t(self):
        return self.time

    @property
    def y(self):
        return self.state


def create_ODE_solver(
    system,
    state_full_init,
    state_init,
    action_init,
    time_start=0.0,
    time_final=10.0,
    max_step=1e-3,
    first_step=1e-6,
    atol=1e-5,
    rtol=1e-3,
    ode_backend="SCIPY",
):
    """
    Create an ODE solver for the given system with the given initial conditions and integration parameters.

    :param system: a system object to be integrated
    :type system: System
    :param state_full_init: the initial state of the system
    :type state_full_init: np.array
    :param state_init: the initial value of the state variable
    :type state_init: np.array
    :param action_init: the initial value of the action variable
    :type action_init: np.array
    :param time_start: the start time of the integration
    :type time_start: float
    :param time_final: the final time of the integration
    :type time_final: float
    :param max_step: the maximum step size to be taken by the integrator
    :type max_step: float
    :param first_step: the step size to be taken by the integrator at the beginning of the integration
    :type first_step: float
    :param atol: the absolute tolerance for the integration
    :type atol: float
    :param rtol: the relative tolerance for the integration
    :type rtol: float
    :param ode_backend: the type of ODE solver to be used, either "SCIPY" or "CASADI"
    :type ode_backend: str
    :return: an ODE solver object
    :rtype: Solver
    """
    if ode_backend == "SCIPY":
        import scipy as sp

        solver = sp.integrate.RK45(
            system.compute_closed_loop_rhs,
            time_start,
            state_full_init,
            time_final,
            max_step=max_step,
            first_step=first_step,
            atol=atol,
            rtol=rtol,
        )

    elif ode_backend == "CASADI":

        integrator = create_CasADi_integrator(system, state_init, action_init, max_step)

        solver = CasADiSolver(
            integrator,
            time_start,
            time_final,
            max_step,
            state_init,
            action_init,
            system,
        )
    return solver


def create_CasADi_integrator(system, state_init, action_init, max_step):
    """
    Create a CasADi integrator for a given system.

    :param system: The system for which to create the integrator.
    :type system: System
    :param state_init: Initial state of the system.
    :type state_init: numpy.ndarray
    :param action_init: Initial action of the system.
    :type action_init: numpy.ndarray
    :param max_step: Maximum step size for the integrator.
    :type max_step: float
    :return: CasADi integrator for the system.
    :rtype: casadi.integrator
    """

    state_symbolic = rc.array_symb(rc.shape(state_init), literal="x")
    action_symbolic = rc.array_symb(rc.shape(action_init), literal="u")
    time = rc.array_symb((1, 1), literal="t")

    ODE = system.compute_dynamics(time, state_symbolic, action_symbolic)
    DAE = {"x": state_symbolic, "p": action_symbolic, "ode": ODE}

    options = {"tf": max_step}

    integrator = casadi.integrator("intg", "rk", DAE, options)

    return integrator
