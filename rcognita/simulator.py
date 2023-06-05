#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains one single class that simulates controller-system (agent-environment) loops.
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
import scipy as sp

import rcognita.base
from .__utilities import rc
from .systems import System, ComposedSystem
from typing import Union, Optional
from abc import ABC, abstractmethod

try:
    import casadi
except ModuleNotFoundError:
    from unittest.mock import MagicMock

    casadi = MagicMock()


class Simulator(rcognita.base.RcognitaBase, ABC):
    def __init__(
        self,
        system: Union[System, ComposedSystem],
        state_init: Optional[np.ndarray] = None,
        time_start: float = 0,
        time_final: float = 1,
        max_step: float = 1e-3,
        first_step: float = 1e-6,
        atol: float = 1e-5,
        rtol: float = 1e-3,
    ):
        self.system = system
        assert state_init, "Initial state for this simulator needs to be passed"
        self.state = state_init
        self.state_init = state_init

        self.time_start = time_start
        self.time_final = time_final
        self.time = time_start

        self.dim_state = state_init.shape[0]
        self.observation = None

        self.max_step = max_step
        self.atol = atol
        self.rtol = rtol

        self.first_step = first_step
        assert hasattr(
            self.system, "system_type"
        ), "System must contain a system_type attribute"

        ## TODO: Add support for other types of systems
        if self.system.system_type == "diff_eqn":
            self.initialize_ode_solver()

    def receive_action(self, action):
        self.system.receive_action(action)

    def do_sim_step(self):
        """
        Do one simulation step and update current simulation data (time, system state and output).

        """

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

    def get_sim_step_data(self):
        return self.time, self.state

    def reset(self):
        if self.system.system_type == "diff_eqn":
            self.initialize_ode_solver()
            self.time = self.time_start
            self.state = self.state_init
            self.system.reset()
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.system.inputs
            )
        else:
            self.time = self.time_start
            self.observation = self.state_full_init

    @abstractmethod
    def initialize_ode_solver(self):
        pass


class SciPy(Simulator):
    def initialize_ode_solver(self):
        self.ODE_solver = sp.integrate.RK45(
            self.system.compute_state_dynamics,
            self.time_start,
            self.state,
            self.time_final,
            max_step=self.max_step,
            first_step=self.first_step,
            atol=self.atol,
            rtol=self.rtol,
        )


class CaADi(Simulator):
    class CasADiSolver:
        def __init__(
            self,
            integrator: casadi.integrator,
            time_start: float,
            time_final: float,
            step_size: float,
            state_init: Union[np.array, casadi.DM],
            action_init: Union[np.array, casadi.DM],
            system: Union[System, ComposedSystem],
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
        self.ODE_solver = self.CasADiSolver(
            self.integrator,
            self.time_start,
            self.time_final,
            self.max_step,
            self.state_init,
            self.system,
        )

    def create_CasADi_integrator(self, system, max_step):
        state_symbolic = rc.array_symb(self.system.dim_state, literal="x")
        action_symbolic = rc.array_symb(self.system.dim_inputs, literal="u")
        time = rc.array_symb((1, 1), literal="t")

        ODE = system.compute_state_dynamics(time, state_symbolic, action_symbolic)
        DAE = {"x": state_symbolic, "p": action_symbolic, "ode": ODE}
        options = {"tf": max_step, "atol": self.atol, "rtol": self.rtol}

        integrator = casadi.integrator("intg", "rk", DAE, options)

        return integrator
