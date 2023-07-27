#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains one single class that simulates controller-system (agent-environment) loops.

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

import rcognita.base
from .__utilities import rc
from .systems import System, ComposedSystem
from typing import Union, Optional
from abc import ABC

try:
    import casadi
except ModuleNotFoundError:
    from unittest.mock import MagicMock

    casadi = MagicMock()


# TODO: DOCSTRING
class Simulator(rcognita.base.RcognitaBase, ABC):
    def __init__(
        self,
        system: Union[System, ComposedSystem],
        state_init: Optional[np.ndarray] = None,
        action_init: Optional[np.ndarray] = None,
        time_start: Optional[float] = 0,
        time_final: Optional[float] = 1,
        max_step: Optional[float] = 1e-3,
        first_step: Optional[float] = 1e-6,
        atol: Optional[float] = 1e-5,
        rtol: Optional[float] = 1e-3,
    ):
        r"""Initialize a simulator.

        Parameters
        ----------
        sys_type : : string
            Type of system by description:

            | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, u, q)`
            | ``discr_fnc`` : difference equation :math:`state^+ = f(state, u, q)`
            | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, u, q)`

        where:

            | :math:`state` : state
            | :math:`u` : input
            | :math:`q` : disturbance

        compute_closed_loop_rhs : : function
            Right-hand side description of the closed-loop system.
            Say, if you instantiated a concrete system (i.e., as an instance of a subclass of ``System`` class with concrete ``compute_closed_loop_rhs`` method) as ``system``,
            this could be just ``system.compute_closed_loop_rhs``.

        sys_out : : function
            System output function.
            Same as above, this could be, say, ``system.out``.

        is_dynamic_controller : : 0 or 1
            If 1, the controller (a.k.a. agent) is considered as a part of the full state vector.

        state_init, disturb_init, action_init : : vectors
            Initial values of the (open-loop) system state, disturbance and input.

        time_start, time_final, sampling_time : : numbers
            Initial, final times and time step size

        max_step, first_step, atol, rtol : : numbers
            Parameters for an ODE solver (used if ``sys_type`` is ``diff_eqn``).

        system : : `System`
            System to be simulated.

        """
        self.system = system
        assert hasattr(
            self.system, "system_type"
        ), "System must contain a system_type attribute"
        if self.system.system_type == "diff_eqn":
            assert (
                state_init is not None
            ), "Initial state for this simulator needs to be passed"

        self.time_start = time_start
        self.time_final = time_final
        if state_init is None:
            self.state_init = self.initialize_init_state()
        else:
            self.state_init = state_init
        if action_init is None:
            self.action_init = self.initialize_init_action()
        else:
            self.action_init = action_init

        self.time = time_start
        self.state = self.state_init
        self.observation = self.get_observation(
            time=self.time, state=self.state_init, inputs=self.action_init
        )

        self.max_step = max_step
        self.atol = atol
        self.rtol = rtol
        self.first_step = first_step

        ## TODO: Add support for other types of systems
        if self.system.system_type == "diff_eqn":
            assert (
                self.time_start is not None and self.time_final is not None
            ), "Must specify time_start and time_final for diff_eqn systems"
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
        return self.time, self.state, self.observation

    def reset(self):
        if self.system.system_type == "diff_eqn":
            self.ODE_solver = self.initialize_ode_solver()
            self.time = self.time_start
            self.state = self.state_init
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.action_init
            )
            self.system.reset()
        else:
            self.time = self.time_start
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.system.inputs
            )

    def initialize_ode_solver(self):
        raise NotImplementedError("Not implemented ODE solver")

    def get_init_state_and_action(self):
        return self.state_init, rc.zeros(self.system.dim_inputs)

    def initialize_init_state(self):
        raise NotImplementedError(
            "Implement this method to initialize the initial state"
            + "if one is intended to be obtained during the runtime"
        )

    def initialize_init_action(self):
        return rc.zeros(self.system.dim_inputs)


class SciPy(Simulator):
    def initialize_ode_solver(self):
        ODE_solver = sp.integrate.RK45(
            self.system.compute_closed_loop_rhs,
            self.time_start,
            self.state,
            self.time_final,
            max_step=self.max_step,
            first_step=self.first_step,
            atol=self.atol,
            rtol=self.rtol,
        )
        return ODE_solver


class CasADi(Simulator):
    class CasADiSolver:
        def __init__(
            self,
            integrator,
            time_start: float,
            time_final: float,
            step_size: float,
            state_init,
            action_init,
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
        assert (
            self.time_start is not None
            and self.time_final is not None
            and self.max_step is not None
        ), (
            "Must specify time_start, time_final and max_step"
            + " in order to initialize CasADi solver"
        )
        ODE_solver = self.CasADiSolver(
            self.integrator,
            self.time_start,
            self.time_final,
            self.max_step,
            self.state_init,
            self.action_init,
            self.system,
        )
        return ODE_solver

    def create_CasADi_integrator(self, system, max_step):
        state_symbolic = rc.array_symb(self.system.dim_state, literal="x")
        action_symbolic = rc.array_symb(self.system.dim_inputs, literal="u")
        time = rc.array_symb((1, 1), literal="t")

        ODE = system.compute_state_dynamics(time, state_symbolic, action_symbolic)
        DAE = {"x": state_symbolic, "p": action_symbolic, "ode": ODE}
        # options = {"tf": max_step, "atol": self.atol, "rtol": self.rtol}
        options = {"tf": max_step}
        integrator = casadi.integrator("intg", "rk", DAE, options)

        return integrator
