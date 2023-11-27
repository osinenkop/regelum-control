#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains one single class that simulates scenario-system (agent-environment) loops.

Contains one single class that simulates scenario-system (agent-environment) loops.
The system can be of three types:

- discrete-time deterministic
- continuous-time deterministic or stochastic
- discrete-time stochastic (to model Markov decision processes)

Remarks:

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np

import regelum
from .utils import rg
from .system import System, ComposedSystem
from typing import Union, Optional
from abc import ABC

try:
    import casadi
except ModuleNotFoundError:
    from unittest.mock import MagicMock

    casadi = MagicMock()


class Simulator(regelum.RegelumBase, ABC):
    """Base class of Simulator."""

    def __init__(
        self,
        system: Union[System, ComposedSystem],
        state_init: Optional[np.ndarray] = None,
        action_init: Optional[np.ndarray] = None,
        time_final: Optional[float] = 1,
        max_step: Optional[float] = 1e-3,
        first_step: Optional[float] = 1e-6,
        atol: Optional[float] = 1e-5,
        rtol: Optional[float] = 1e-3,
    ):
        """Initialize an instance of Simulator.

        Args:
            system (Union[System, ComposedSystem]): A controlled system
                to be simulated
            state_init (Optional[np.ndarray], optional): Set initial
                state manually, defaults to None
            action_init (Optional[np.ndarray], optional): Set initial
                action manually, defaults to None
            time_final (Optional[float], optional): Time at which
                simulation ends, defaults to 1
            max_step (Optional[float], optional): Total duration of one
                simulation step, defaults to 1e-3
            first_step (Optional[float], optional): Used with
                integrators with changing simulation steps, defaults to
                1e-6
            atol (Optional[float], optional): Absolute tollerance of
                integrator, defaults to 1e-5
            rtol (Optional[float], optional): Relative tollerance of
                integrator, defaults to 1e-3
        """
        self.system = system
        assert hasattr(
            self.system, "system_type"
        ), "System must contain a system_type attribute"
        if self.system.system_type == "diff_eqn":
            assert (
                state_init is not None
            ), "Initial state for this simulator needs to be passed"

        self.time_final = time_final
        if state_init is None:
            self.state_init = self.initialize_init_state()
        else:
            self.state_init = state_init
        if action_init is None:
            self.action_init = self.initialize_init_action()
        else:
            self.action_init = action_init

        self.time = 0.0
        self.state = self.state_init
        self.observation = self.get_observation(
            time=self.time, state=self.state_init, inputs=self.action_init
        )

        self.max_step = max_step
        self.atol = atol
        self.rtol = rtol
        self.first_step = first_step

        if self.system.system_type == "diff_eqn":
            self.ODE_solver = self.initialize_ode_solver()

    def receive_action(self, action):
        self.system.receive_action(action)

    def get_observation(self, time, state, inputs):
        return self.system.get_observation(time, state, inputs)

    def do_sim_step(self):
        """Do one simulation step and update current simulation data (time, system state and output)."""
        if self.system.system_type == "diff_eqn":
            try:
                self.ODE_solver.step()
            except RuntimeError:
                self.reset()
                return -1

            self.time = self.ODE_solver.t
            self.state = self.ODE_solver.y
            self.observation = self.get_observation(
                time=self.time, state=self.state, inputs=self.system.inputs
            )
        else:
            raise ValueError("Invalid system description")

    @apply_callbacks()
    def get_sim_step_data(self):
        return self.time, self.state, self.observation, self.get_simulation_metadata()

    def get_simulation_metadata(self):
        ...

    def reset(self):
        if self.system.system_type == "diff_eqn":
            self.ODE_solver = self.initialize_ode_solver()
            self.time = 0.0
            self.state = self.state_init
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.action_init
            )
        else:
            self.time = 0.0
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.system.inputs
            )

    def initialize_ode_solver(self):
        raise NotImplementedError("Not implemented ODE solver")

    def get_init_state_and_action(self):
        return self.state_init, rg.zeros((1, self.system.dim_inputs))

    def initialize_init_state(self):
        raise NotImplementedError(
            "Implement this method to initialize the initial state"
            + "if one is intended to be obtained during the runtime"
        )

    def initialize_init_action(self):
        return rg.zeros(self.system.dim_inputs)


class SciPy(Simulator):
    """Class for SciPy integrators."""

    def initialize_ode_solver(self):
        import scipy as sp

        ODE_solver = sp.integrate.RK45(
            self.system.compute_closed_loop_rhs,
            self.state,
            self.time_final,
            max_step=self.max_step,
            first_step=self.first_step,
            atol=self.atol,
            rtol=self.rtol,
        )
        return ODE_solver


class CasADi(Simulator):
    """Class for CasADi integrators."""

    class CasADiSolver:
        """Nested class to wrap casadi integrator into a uniform API."""

        def __init__(
            self,
            integrator,
            time_final: float,
            step_size: float,
            state_init,
            action_init,
            system: Union[System, ComposedSystem],
        ):
            """Initialize a CasADiSolver object.

            Args:
                integrator (casadi.integrator): A CasADi integrator
                    object.
                time_final (float): The final time for the solver.
                step_size (float): The step size for the solver.
                state_init (np.array): The initial state for the solver.
                action_init (np.array): The initial action for the
                    solver.
                system (System): The system object for the solver.
            """
            self.integrator = integrator
            self.time_final = time_final
            self.step_size = step_size
            self.time = 0.0
            self.state_init = state_init
            self.state = self.state_init
            self.state_new = self.state
            self.action_init = action_init
            self.action = self.action_init
            self.system = system

        def step(self):
            """Advance the solver by one step."""
            if self.time >= self.time_final:
                raise RuntimeError("An attempt to step with a finished solver")
            state_new = np.squeeze(
                self.integrator(x0=self.state, p=self.system.inputs)["xf"].full()
            )
            self.time += self.step_size
            self.state = state_new

        @property
        def t(self):
            return self.time

        @property
        def y(self):
            return self.state

    def initialize_ode_solver(self):
        self.integrator = self.create_CasADi_integrator(self.system, self.max_step)
        assert self.time_final is not None and self.max_step is not None, (
            "Must specify time_final and max_step"
            + " in order to initialize CasADi solver"
        )
        ODE_solver = self.CasADiSolver(
            self.integrator,
            self.time_final,
            self.max_step,
            self.state_init,
            self.action_init,
            self.system,
        )
        return ODE_solver

    def create_CasADi_integrator(self, system, max_step):
        state_symbolic = rg.array_symb(self.system.dim_state, literal="x")
        action_symbolic = rg.array_symb(self.system.dim_inputs, literal="u")
        time = rg.array_symb((1, 1), literal="t")

        ODE = system.compute_state_dynamics(
            time, state_symbolic, action_symbolic, _native_dim=True
        )
        DAE = {"x": state_symbolic, "p": action_symbolic, "ode": ODE}

        # options = {"tf": max_step, "atol": self.atol, "rtol": self.rtol}
        options = {"tf": max_step}
        integrator = casadi.integrator("intg", "rk", DAE, options)

        # integrator = casadi.integrator("intg", "rk", DAE, 0, max_step)

        return integrator
