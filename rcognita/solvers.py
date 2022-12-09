from abc import ABC, abstractmethod
import warnings

try:
    import casadi
except ModuleNotFoundError:
    warnings.warn_explicit(
        "\n CasADiSolver is not available",
        UserWarning,
        __file__,
        42,
    )

import numpy as np
from .systems import System


class Solver(ABC):
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
        pass


class CasADiSolver(Solver):
    def __init__(
        self,
        integrator: casadi.integrator,
        time_start: float,
        time_final: float,
        step_size: float,
        state_init: np.array,
        action_init: np.array,
        system: System,
    ):

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
    self,
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
    ode_solver="NUMPY",
):

    if ode_solver == "NUMPY":
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

    elif ode_solver == "CASADI":

        integrator = create_CasADi_integrator(
            system.compute_dynamics, state_init, action_init, max_step
        )

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


def create_CasADi_integrator(self, system, state_init, action_init, max_step):
    state_symbolic = self.array_symb(self.shape(state_init), literal="x")
    action_symbolic = self.array_symb(self.shape(action_init), literal="u")
    time = self.array_symb((1, 1), literal="t")

    ODE = system.compute_dynamics(time, state_symbolic, action_symbolic)
    DAE = {"x": state_symbolic, "p": action_symbolic, "ode": ODE}

    options = {"tf": max_step}

    integrator = casadi.integrator("intg", "rk", DAE, options)

    return integrator
