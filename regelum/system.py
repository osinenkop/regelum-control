"""Contains a generic interface for systems (environments) as well as concrete systems as realizations of the former.

Remarks:

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""
from __future__ import annotations
import numpy as np


import regelum
from abc import ABC, abstractmethod
from .__utilities import rg
from typing import Optional, Union, List
from typing_extensions import Self


class SystemInterface(regelum.RegelumBase, ABC):
    """
    Generic interface for systems (environments).

    Attributes:
        _name (Optional[str]): The name of the system.
        _system_type (Optional[str]): The type of the system. For example, "diff_eqn" for a differential equation system.
        _dim_state (Optional[int]): The dimensionality of the system's state.
        _dim_inputs (Optional[int]): The dimensionality of the system's inputs. Inputs mostly are interpreted as control inputs.
        _dim_observation (Optional[int]): The dimensionality of the system's observation.
        _parameters (Optional[dict]): The parameters of the system.
        _observation_naming (Optional[List[str]]): The naming of the observation dimensions.
        _inputs_naming (Optional[List[str]]): The naming of the inputs dimensions.
        _action_naming (Optional[List[str]]): The naming of the action dimensions.
        _action_bounds (Optional[List[List[float]]]): The bounds for each action dimension.
    """

    _name: Optional[str] = None
    _system_type: Optional[str] = None
    _dim_state: Optional[int] = None
    _dim_inputs: Optional[int] = None
    _dim_observation: Optional[int] = None
    _parameters: Optional[dict] = {}
    _observation_naming: Optional[List[str]] = None
    _states_naming: Optional[List[str]] = None
    _inputs_naming: Optional[List[str]] = None
    _action_bounds: Optional[List[List[float]]] = None

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("system._name should be set")
        return self._name

    @property
    def system_type(self) -> str:
        if self._system_type is None:
            raise ValueError("system._system_type should be set")
        return self._system_type

    @property
    def dim_state(self) -> int:
        if self._dim_state is None:
            raise ValueError("system._dim_state should be set")
        return self._dim_state

    @property
    def dim_observation(self) -> int:
        if self._dim_observation is None:
            raise ValueError("system._dim_observation should be set")
        return self._dim_observation

    @property
    def dim_inputs(self):
        if self._dim_inputs is None:
            raise ValueError("system._dim_inputs should be set")
        return self._dim_inputs

    @property
    def parameters(self):
        return self._parameters

    @property
    def state_naming(self):
        return self._state_naming

    @property
    def observation_naming(self):
        return self._observation_naming

    @property
    def inputs_naming(self):
        return self._inputs_naming

    @property
    def action_bounds(self):
        return self._action_bounds

    @abstractmethod
    def _compute_state_dynamics(time, state, inputs):
        """Compute the dynamics of the state at a given time.

        Parameters:
            time (float): The time at which to compute the state dynamics.
            state (object): The current state of the system.
            inputs (object): The inputs to the system, e.g. control inputs.

        Returns:


        """
        pass

    def compute_state_dynamics(
        self, time, state, inputs, _native_dim=False
    ) -> np.ndarray:
        if not _native_dim:
            return rg.force_row(
                self._compute_state_dynamics(
                    time, rg.force_column(state), rg.force_column(inputs)
                )
            )
        else:
            return self._compute_state_dynamics(time, state, inputs)

    @abstractmethod
    def _get_observation(self, time, state, inputs):
        pass

    def get_observation(self, time, state, inputs, _native_dim=False, is_batch=False):
        if not is_batch:
            if not _native_dim:
                return rg.force_row(
                    self._get_observation(
                        time, rg.force_column(state), rg.force_column(inputs)
                    )
                )
            else:
                return self._get_observation(time, state, inputs)
        else:
            observations = rg.zeros(state.shape, prototype=state)
            for i in range(state.shape[0]):
                observations[i, :] = self.get_observation(
                    time=time,
                    state=state[i, :],
                    inputs=inputs[i, :],
                    is_batch=False,
                )
            return observations

    def apply_action_bounds(self, action: np.array):
        return (
            np.clip(
                action,
                np.array(self.action_bounds)[:, 0],
                np.array(self.action_bounds)[:, 1],
            )
            if self.action_bounds is not None
            else action
        )


class ComposedSystem(SystemInterface):
    """Base class for composed systems.

    An instance of this class is being created automatically when applying a `@` operation on two systems.
    """

    def __init__(
        self,
        sys_left: Union[System, Self],
        sys_right: Union[System, Self],
        io_mapping: Optional[list] = None,
        output_mode="right",
        state_naming=None,
        inputs_naming=None,
        observation_naming=None,
        action_bounds=None,
    ):
        """Initialize a composed system by specifying systems to compose.

        Args:
            sys_left (Union[System, Self]): System outputs of which are
                to connected to the inputs of the right system
            sys_right (Union[System, Self]): Second system that can be
                connected to the inputs of the left system
            io_mapping (Optional[list], optional): Mapping of inputs of
                the right system to inputs of the left system, defaults
                to None
            output_mode (str, optional): How to combine the result
                outputs, defaults to "right"
        """
        self._state_naming = state_naming
        self._inputs_naming = inputs_naming
        self._observation_naming = observation_naming
        self._action_bounds = action_bounds

        if io_mapping is None:
            io_mapping = np.arange(min(sys_left.dim_state, sys_right.dim_inputs))

        assert output_mode in [
            "state",
            "right",
            "both",
        ], "output_mode must be 'state', 'right' or 'both'"

        if "diff_eqn" in [sys_left.system_type, sys_right.system_type]:
            self._system_type = "diff_eqn"
        else:
            self._system_type = sys_right.system_type

        self.sys_left = sys_left
        self.sys_right = sys_right
        self._parameters = sys_left.parameters | sys_right.parameters
        self._dim_state = self.sys_right.dim_state + self.sys_left.dim_state
        if output_mode == "state":
            self._dim_observation = self.sys_left.dim_state + self.sys_right.dim_state
        elif output_mode == "right":
            self._dim_observation = self.sys_right.dim_observation
        elif output_mode == "both":
            self._dim_observation = (
                self.sys_left.dim_observation + self.sys_right.dim_observation
            )
        self.rout_idx, self.occupied_idx = self.__get_routing(io_mapping)
        self._dim_inputs = (
            self.sys_right.dim_inputs
            + self.sys_left.dim_inputs
            - len(self.occupied_idx)
        )
        self._name = self.sys_left.name + " + " + self.sys_right.name

        self.free_right_input_indices = np.setdiff1d(
            np.arange(self.sys_right.dim_inputs).astype(int),
            self.occupied_idx.astype(int),
        )
        self.output_mode = output_mode
        self.forward_permutation = np.arange(self.dim_observation).astype(int)
        self.inverse_permutation = np.arange(self.dim_observation).astype(int)

    @staticmethod
    def __get_routing(io_mapping):
        io_mapping_extended = []

        for i, outputs in enumerate(io_mapping):
            assert type(outputs) in [
                tuple,
                int,
                np.int32,
                np.int64,
                None,
            ], (
                "io_mapping must be a list of ints or tuples or Nones. "
                + f"However a value of type {type(outputs)} was provided."
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            elif outputs is None:
                continue
            for output in outputs:
                io_mapping_extended.append([i, output])

        io_mapping_extended = sorted(io_mapping_extended, key=lambda x: x[1])
        rout_idx, occupied_idx = rg.array(io_mapping_extended).astype(int).T
        return rout_idx, occupied_idx

    def _compute_state_dynamics(self, time, state, inputs, _native_dim=True):
        state = rg.array(state, prototype=state)
        inputs = rg.array(inputs, prototype=state)
        state = state[self.forward_permutation]

        state_for_left, state_for_right = (
            state[: self.sys_left.dim_state],
            state[self.sys_left.dim_state :],
        )
        inputs_for_left = inputs[: self.sys_left.dim_inputs]
        dstate_of_left = rg.squeeze(
            self.sys_left.compute_state_dynamics(
                time=time,
                state=state_for_left,
                inputs=inputs_for_left,
                _native_dim=_native_dim,
            )
        )
        outputs_of_left = rg.squeeze(
            self.sys_left.get_observation(
                time=time,
                state=state_for_left,
                inputs=inputs_for_left,
                _native_dim=_native_dim,
            )
        )

        inputs_for_right = rg.zeros(
            self.sys_right.dim_inputs,
            prototype=(state, inputs),
        )
        inputs_for_right[self.occupied_idx] = outputs_of_left[self.rout_idx]
        inputs_for_right[self.free_right_input_indices] = rg.reshape(
            inputs[self.sys_left.dim_inputs :],
            rg.shape(inputs_for_right[self.free_right_input_indices]),
        )

        dstate_of_right = rg.squeeze(
            self.sys_right.compute_state_dynamics(
                time=time,
                state=state_for_right,
                inputs=inputs_for_right,
                _native_dim=_native_dim,
            )
        )
        final_dstate_vector = rg.hstack((dstate_of_left, dstate_of_right))

        assert (
            final_dstate_vector is not None
        ), f"final dstate_vector of system {self.name} is None"
        final_dstate_vector = final_dstate_vector[self.inverse_permutation]
        if not _native_dim:
            final_dstate_vector = rg.force_row(final_dstate_vector)
        return final_dstate_vector

    def _get_observation(self, time, state, inputs, _native_dim=False):
        state = rg.array(state, prototype=state)
        inputs = rg.array(inputs, prototype=state)

        inputs_for_left = inputs[: self.sys_left.dim_inputs]
        state_for_left, state_for_right = (
            state[: self.sys_left.dim_state],
            state[self.sys_left.dim_state :],
        )
        outputs_of_left = rg.squeeze(
            self.sys_left.get_observation(
                time=time,
                state=state_for_left,
                inputs=inputs_for_left,
                _native_dim=_native_dim,
            )
        )

        inputs_for_right = rg.zeros(
            self.sys_right.dim_inputs,
            prototype=(state, inputs),
        )
        inputs_for_right[self.occupied_idx] = outputs_of_left[self.rout_idx]
        inputs_for_right[self.free_right_input_indices] = rg.reshape(
            inputs[self.sys_left.dim_inputs :],
            rg.shape(inputs_for_right[self.free_right_input_indices]),
        )
        outputs_of_right = self.sys_right.get_observation(
            time=time,
            state=state_for_right,
            inputs=inputs_for_right,
            _native_dim=_native_dim,
        )
        if self.output_mode == "right":
            output = outputs_of_right
        elif self.output_mode == "state":
            output = state
        elif self.output_mode == "both":
            output = rg.concatenate((state_for_left, state_for_right))
        else:
            raise NotImplementedError

        ## TODO: implement 'preserve' mode

        return output

    def receive_action(self, action):
        self.inputs = action

    def update_system_parameters(self, inputs):
        assert isinstance(inputs, dict)
        self.sys_left.update_system_parameters(inputs)
        self.sys_right.update_system_parameters(inputs)

    def permute_state(self, permutation: Union[list, np.array]) -> Self:
        """Permute an order at which the system outputs are returned.

        Args:
            permutation (Union[list, np.array]): Permutation represented
                as an array of indices

        Returns:
            Self: link to self
        """
        self.forward_permutation = rg.array(permutation).astype(int)
        self.inverse_permutation = self.get_inverse_permutation(permutation)
        return self

    def get_inverse_permutation(self, permutation):
        self.current_permutation = permutation
        permutation = np.asanyarray(permutation)
        inverse_permutation = np.empty_like(permutation)
        inverse_permutation[permutation] = np.arange(permutation.size)
        return inverse_permutation

    def compose(self, sys_right, io_mapping=None, output_mode="state"):
        return ComposedSystem(
            self, sys_right, io_mapping=io_mapping, output_mode=output_mode
        )

    def __matmul__(self, sys_right):
        return self.compose(sys_right)


class System(SystemInterface):
    """Base class for controlled systems implementation."""

    def __init__(
        self,
        system_parameters_init=None,
        state_init: Optional[np.ndarray] = None,
        inputs_init: Optional[np.ndarray] = None,
    ):
        """Initialize an instance of a system.

        Args:
            system_parameters_init (dict, optional): Set system
                parameters manually, defaults to {}
            state_init (Optional[np.ndarray], optional): Set initial
                state manually, defaults to None
            inputs_init (Optional[np.ndarray], optional): Set initial
                inputs manually, defaults to None
        """
        if system_parameters_init is None:
            system_parameters_init = {}
        assert self.name is not None
        assert self.system_type is not None
        assert self.dim_state is not None
        assert self.dim_inputs is not None
        assert self.dim_observation is not None
        assert isinstance(
            system_parameters_init, dict
        ), "system_parameters_init should be a dict"

        if system_parameters_init:
            self._parameters.update(system_parameters_init)

        self.system_parameters_init = self._parameters

        if state_init is None:
            self.state = rg.zeros(self.dim_state)
        else:
            self.state = state_init

        if inputs_init is None:
            self.inputs = rg.zeros(self.dim_inputs)
        else:
            self.inputs = inputs_init

    def _get_observation(self, time, state, inputs):
        return state

    def receive_action(self, action):
        self.inputs = action

    def update_system_parameters(self, inputs):
        assert isinstance(inputs, dict)
        self._parameters.update(inputs)
        return self.parameters

    def compose(self, sys_right, io_mapping=None, output_mode="state"):
        return ComposedSystem(
            self, sys_right, io_mapping=io_mapping, output_mode=output_mode
        )

    def __matmul__(self, sys_right):
        return self.compose(sys_right)


class KinematicPoint(System):
    """System representing Kinematic Point (omnibot)."""

    _name = "kinematic-point"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _observation_naming = _state_naming = ["x", "y"]
    _inputs_naming = ["v_x", "v_y"]
    _action_bounds = [[-10.0, 10.0], [-10.0, 10.0]]

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        for i in range(rg.shape(inputs)[0]):
            Dstate[i] = inputs[i]

        return Dstate


class InvertedPendulumPID(System):
    """System class: mathematical pendulum."""

    _name = "inverted-pendulum"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 1
    _dim_observation = 3
    _parameters = {"m": 1, "g": 9.8, "l": 1}
    _observation_naming = _state_naming = ["angle", "angular velocity"]
    _inputs_naming = ["momentum"]
    _action_bounds = [[-20.0, 20.0]]

    def __init__(self, *args, **kwargs):
        """Initialize an instance of an Inverted Pendulum, which gives an observation suitable for PID scenario."""
        super().__init__(*args, **kwargs)

        self.time_old = 0
        self.integral_alpha = 0

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, g, l = (
            self.parameters["m"],
            self.parameters["g"],
            self.parameters["l"],
        )

        Dstate[0] = state[1]
        Dstate[1] = g / l * rg.sin(state[0]) + inputs[0] / (m * l**2)

        return Dstate

    def _get_observation(self, time, state, inputs):
        delta_time = time - self.time_old if time is not None else 0
        self.integral_alpha += delta_time * state[0]

        return rg.hstack([state[0], self.integral_alpha, state[1]])

    def reset(self):
        self.time_old = 0
        self.integral_alpha = 0


class InvertedPendulumPD(InvertedPendulumPID):
    """System class: ordinary mathematical pendulum."""

    _dim_observation = 2

    def _get_observation(self, time, state, inputs):
        return rg.hstack([state[0], state[1]])


class ThreeWheeledRobot(System):
    r"""System class: 3-wheeled robot with dynamical actuators.

    Description
    -----------
    Three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_]

    .. math::
        \begin{array}{ll}
                        \dot x_с & = v \cos \angle \newline
                        \dot y_с & = v \sin \angle \newline
                        \dot \angle & = \omega \newline
                        \dot v & = \left( \frac 1 m F + q_1 \right) \newline
                        \dot \omega & = \left( \frac 1 I M + q_2 \right)
        \end{array}

    **Variables**

    | :math:`x_с` : state-coordinate [m]
    | :math:`y_с` : observation-coordinate [m]
    | :math:`\angle` : turning angle [rad]
    | :math:`v` : speed [m/s]
    | :math:`\omega` : revolution speed [rad/s]
    | :math:`F` : pushing force [N]
    | :math:`M` : steering torque [Nm]
    | :math:`m` : robot mass [kg]
    | :math:`I` : robot moment of inertia around vertical axis [kg m\ :sup:`2`]
    | :math:`disturb` : actuator disturbance (see :func:`~RLframe.system.disturbDyn`). Is zero if ``is_disturb = 0``

    :math:`state = [x_c, y_c, \angle, v, \omega]`

    :math:`inputs = [F, M]`

    ``pars`` = :math:`[m, I]`

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
        nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    """

    _name = "three-wheeled-robot"
    _system_type = "diff_eqn"
    _dim_state = 5
    _dim_inputs = 2
    _dim_observation = 5
    _parameters = {"m": 10, "I": 1}
    _observation_naming = _state_naming = [
        "x",
        "y",
        "angle",
        "l_velocity",
        "angular_velocity",
    ]
    _inputs_naming = ["Force", "Momentum"]
    _action_bounds = [[-25.0, 25.0], [-5.0, 5.0]]

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, I = self.parameters["m"], self.parameters["I"]

        Dstate[0] = state[3] * rg.cos(state[2])
        Dstate[1] = state[3] * rg.sin(state[2])
        Dstate[2] = state[4]
        Dstate[3] = 1 / m * inputs[0]
        Dstate[4] = 1 / I * inputs[1]

        return Dstate


class Integrator(System):
    """System yielding Non-holonomic double integrator when composed with kinematic thre-wheeled robot."""

    _name = "integral-parts"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _parameters = {"m": 10, "I": 1}

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, I = self.parameters["m"], self.parameters["I"]

        Dstate[0] = 1 / m * inputs[0]
        Dstate[1] = 1 / I * inputs[1]

        return Dstate


class ThreeWheeledRobotNI(System):
    r"""Implements the ThreeWheeledRobotNI (Non-holonomic robot a.k.a. Brockett integrator).

    This system class defines the dynamics of a 3-wheeled robot with non-holonomic constraints.
    The robot's dynamics are given by the following differential equations:

    .. math::
        \begin{aligned}
            &  \dot{x}_{\text{rob}} = v \cos(\\vartheta), \\
            &  \dot{y}_{\text{rob}} = v \sin(\\vartheta), \\
            & \dot{\\vartheta} = \\mega.
        \end{aligned}

    Where:
    - :math:`\dot{x}_{\text{rob}}` is the rate of change of the robot's x-position.
    - :math:`\dot{y}_{\text{rob}}` is the rate of change of the robot's y-position.
    - :math:`\vartheta` is the robot's orientation.
    - :math:`v` is the linear velocity input.
    - :math:`\omega` is the angular velocity input.
    """

    _name = "three-wheeled-robot-ni"
    _system_type = "diff_eqn"
    _dim_state = 3
    _dim_inputs = 2
    _dim_observation = 3
    _observation_naming = _state_naming = ["x", "y", "angle"]
    _inputs_naming = ["velocity", "angular velocity"]
    _action_bounds = [[-25.0, 25.0], [-5.0, 5.0]]

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(self.dim_state, prototype=(state, inputs))

        Dstate[0] = inputs[0] * rg.cos(state[2])
        Dstate[1] = inputs[0] * rg.sin(state[2])
        Dstate[2] = inputs[1]

        return Dstate


ThreeWheeledRobotComposed = (Integrator() @ ThreeWheeledRobotNI()).permute_state(
    [3, 4, 0, 1, 2]
)


class TwoTank(System):
    """Two-tank system with nonlinearity."""

    _name = "two-tank"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 1
    _dim_observation = 2
    _parameters = {"tau1": 18.4, "tau2": 24.4, "K1": 1.3, "K2": 1.0, "K3": 0.2}
    _observation_naming = _state_naming = ["h1", "h2"]
    _inputs_naming = ["P"]
    _action_bounds = [[0.0, 1.0]]

    def _compute_state_dynamics(self, time, state, inputs):
        tau1, tau2, K1, K2, K3 = (
            self.parameters["tau1"],
            self.parameters["tau2"],
            self.parameters["K1"],
            self.parameters["K2"],
            self.parameters["K3"],
        )

        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )
        Dstate[0] = 1 / (tau1) * (-state[0] + K1 * inputs[0])
        Dstate[1] = 1 / (tau2) * (-state[1] + K2 * state[0] + K3 * state[1] ** 2)

        return Dstate


class ConstantReference(System):
    """Subtracts reference from system."""

    name = "constant_reference"
    _system_type = "diff_eqn"
    _dim_state = 0
    _dim_inputs = 2
    _dim_observation = 2
    _parameters = {"reference": np.array([[0.4], [0.4]])}

    def __init__(self, reference: Optional[Union[List[float], np.array]] = None):
        """Instantiate ConstantReference.

        Args:
            reference (Optional[Union[List[float], np.array]], optional):
                reference to be substracted from inputs, defaults to
                None
        """
        if reference is None:
            super().__init__(
                system_parameters_init=None, state_init=None, inputs_init=None
            )
        else:
            super().__init__(
                system_parameters_init={
                    "reference": np.array(reference).reshape(-1, 1)
                },
                state_init=None,
                inputs_init=None,
            )
            self._dim_inputs = self._dim_observation = (
                np.array(reference).reshape(-1).shape[0]
            )

    def _get_observation(self, time, state, inputs):
        return inputs - rg.array(
            self.parameters["reference"], prototype=inputs, _force_numeric=True
        )

    def _compute_state_dynamics(self, time, state, inputs):
        return inputs


class SystemWithConstantReference(ComposedSystem):
    """Creates system with that substracts from state a reference value."""

    def __init__(self, system: System, state_reference: Union[List[float], np.array]):
        """Instantiate System with ConstantReference.

        The result ComposedSystem's method get_observation subtracts from state reference value state_reference.

        Args:
            system (System): system
            state_reference (Union[List[float], np.array]): reference to
                be subtracted from state.
        """
        constant_reference = ConstantReference(state_reference)

        assert (
            system.dim_state == constant_reference.dim_inputs
        ), "state_reference should have the same length as dimension of state of system"

        super().__init__(
            sys_left=system,
            sys_right=constant_reference,
            io_mapping=None,
            output_mode="right",
            inputs_naming=system.inputs_naming,
            state_naming=system.state_naming,
            observation_naming=[s + "-ref" for s in system.state_naming],
            action_bounds=system.action_bounds,
        )


class GridWorld(System):
    """A simple 2-dimensional grid world with five actions: left, right, up, down and do nothing.

    The inputs encoding rule is as follows: right, left, up, down, do nothing -> 0, 1, 2, 3, 4.
    """

    def __init__(self, dims, terminal_state):
        """Initialize an instance of GridWorld.

        Args:
            dims (tuple): grid dimensions (height, width)
            terminal_state (list): coordinates of goal cell
        """
        self.dims = dims
        self.terminal_state = terminal_state

    def _compute_state_dynamics(self, current_state, inputs):
        if tuple(self.terminal_state) == tuple(current_state):
            return current_state
        if inputs == 0:
            if current_state[1] < self.dims[1] - 1:
                return (current_state[0], current_state[1] + 1)
        elif inputs == 2:
            if current_state[0] > 0:
                return (current_state[0] - 1, current_state[1])
        elif inputs == 1:
            if current_state[1] > 0:
                return (current_state[0], current_state[1] - 1)
        elif inputs == 3:
            if current_state[0] < self.dims[0] - 1:
                return (current_state[0] + 1, current_state[1])
        return current_state


class CartPole(System):
    """Cart pole system without friction."""

    _name = "cartpole"
    _system_type = "diff_eqn"
    _dim_state = 4
    _dim_inputs = 1
    _dim_observation = 4
    _parameters = {"m_c": 0.1, "m_p": 2.0, "g": 9.81, "l": 0.5}
    _observation_naming = _state_naming = ["angle", "x", "angle_dot", "x_dot"]
    _inputs_naming = ["force"]
    _action_bounds = [[-300.0, 300.0]]

    def _compute_state_dynamics(self, time, state, inputs, disturb=None):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m_c, m_p, g, l = (
            self.parameters["m_c"],
            self.parameters["m_p"],
            self.parameters["g"],
            self.parameters["l"],
        )
        theta = state[0]
        theta_dot = state[2]
        x_dot = state[3]

        sin_theta = rg.sin(theta)
        cos_theta = rg.cos(theta)

        Dstate[0] = theta_dot

        Dstate[1] = x_dot

        Dstate[2] = (
            (
                g * sin_theta
                - cos_theta
                * (inputs[0] + m_p * l * theta_dot**2 * sin_theta)
                / (m_c + m_p)
            )
            / l
            / (4 / 3 - m_p * (cos_theta**2) / (m_c + m_p))
        )
        Dstate[3] = (
            inputs[0] + m_p * l * (theta_dot**2 * sin_theta - Dstate[0] * cos_theta)
        ) / (m_c + m_p)

        return Dstate

    def _get_observation(self, time, state, inputs):
        theta = state[0]
        x = state[1]
        theta_dot = state[2]
        x_dot = state[3]

        theta_observed = theta - rg.floor(theta / (2 * np.pi)) * 2 * np.pi
        theta_observed = rg.if_else(
            theta_observed > np.pi,
            theta_observed - 2 * np.pi,
            theta - rg.floor(theta / (2 * np.pi)) * 2 * np.pi,
        )

        return rg.hstack([theta_observed, x, theta_dot, x_dot])


class LunarLander(System):
    """Lunar lander system.

    link: https://web.aeromech.usyd.edu.au/AMME3500/Course_documents/material/tutorials/Assignment%204%20Lunar%20Lander%20Solution.pdf.
    """

    _name = "lander"
    _system_type = "diff_eqn"
    _dim_state = 6
    _dim_inputs = 2
    _dim_observation = 6
    _parameters = {"m": 10, "J": 3.0, "g": 1.625, "a": 1.0, "r": 1.0, "sigma": 1.0}
    _observation_naming = _state_naming = [
        "x",
        "y",
        "theta",
        "x_dot",
        "y_dot",
        "theta_dot",
    ]
    _inputs_naming = ["vertical force", "side force"]
    _action_bounds = [[-100.0, 100.0], [-50.0, 50.0]]

    def __init__(self, *args, **kwargs):
        """Initialize an instance of LunarLander by specifying relevant physical parameters."""
        super().__init__(*args, **kwargs)
        self.alpha = np.arctan(self.parameters["a"] / self.parameters["r"])
        self.l = np.sqrt(self.parameters["a"] ** 2 + self.parameters["r"] ** 2)
        self.is_landed = False

    def _compute_state_dynamics(self, time, state, inputs, disturb=None):
        Dstate_before_landing = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, J, g = (
            self.parameters["m"],
            self.parameters["J"],
            self.parameters["g"],
        )

        theta = state[2]
        x_dot = state[3]
        y_dot = state[4]
        theta_dot = state[5]

        left_support, right_support = self.compute_supports_geometry(state[:2], theta)

        self.is_landed = (
            rg.if_else(left_support[1] <= 0, 1, 0)
            + rg.if_else(right_support[1] <= 0, 1, 0)
        ) > 0

        F_l = inputs[0] * (1 - self.is_landed)
        F_t = inputs[1] * (1 - self.is_landed)

        self.is_landed_left = rg.if_else(left_support[1] <= 0, 1, 0)
        self.is_landed_right = rg.if_else(right_support[1] <= 0, 1, 0)
        self.is_landed_vertex = rg.if_else(state[1] <= 0, 1, 0)
        self.is_freezed = (
            self.is_landed_left * self.is_landed_right + self.is_landed_vertex
        ) > 0
        self.is_landed = (
            self.is_landed_left + self.is_landed_right + self.is_landed_vertex
        ) > 0

        Dstate_before_landing[0] = x_dot
        Dstate_before_landing[1] = y_dot
        Dstate_before_landing[2] = theta_dot
        Dstate_before_landing[3] = 1 / m * (F_l * rg.cos(theta) - F_t * rg.sin(theta))
        Dstate_before_landing[4] = (
            1 / m * (F_l * rg.sin(theta) + F_t * rg.cos(theta)) - g
        )
        Dstate_before_landing[5] = (4 * F_l) / J

        Dstate_landed_right = self._compute_pendulum_dynamics(
            angle=-theta - self.alpha,
            angle_dot=theta_dot,
            prototype=(state, inputs),
        )

        Dstate_landed_left = self._compute_pendulum_dynamics(
            angle=self.alpha - theta,
            angle_dot=theta_dot,
            prototype=(state, inputs),
        )

        # Check if any of the two lander's supports touched the ground. If yes, freeze the state.

        Dstate = (1 - self.is_freezed) * (
            (1 - self.is_landed) * Dstate_before_landing
            + self.is_landed
            * (
                self.is_landed_right * Dstate_landed_right
                + self.is_landed_left * Dstate_landed_left
            )
        )

        return Dstate

    def _compute_pendulum_dynamics(self, angle, angle_dot, prototype):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=prototype,
        )
        g = self.parameters["g"]

        x = self.l * rg.sin(angle)
        y = self.l * rg.cos(angle)
        angular_acceleration = g / self.l**2 * x
        Dstate[5] = angular_acceleration
        Dstate[0] = angle_dot * y
        Dstate[1] = -angle_dot * x
        Dstate[2] = -angle_dot
        Dstate[3] = y * angular_acceleration - angle_dot**2 * x
        Dstate[4] = -x * angular_acceleration - angle_dot**2 * y

        return Dstate

    def compute_supports_geometry(self, xi, theta):
        A = rg.zeros((2, 2), prototype=xi)
        xi_2 = rg.zeros((2, 1), prototype=xi)
        xi_3 = rg.zeros((2, 1), prototype=xi)

        A[0, 0] = rg.cos(theta)
        A[0, 1] = -rg.sin(theta)
        A[1, 0] = rg.sin(theta)
        A[1, 1] = rg.cos(theta)

        a, r = self.parameters["a"], self.parameters["r"]
        xi_2[0, 0] = xi[0, 0] - a
        xi_2[1, 0] = xi[1, 0] - r
        xi_3[0, 0] = xi[0, 0] + a
        xi_3[1, 0] = xi[1, 0] - r

        xi_2_d = xi_2 - xi
        xi_3_d = xi_3 - xi

        xi_2_d_rot = A @ xi_2_d
        xi_3_d_rot = A @ xi_3_d
        xi_2_new = xi + xi_2_d_rot
        xi_3_new = xi + xi_3_d_rot
        return xi_2_new, xi_3_new

    def compute_reaction(
        self,
        r,
        r_support,
    ):
        m, g, sigma = (
            self.parameters["m"],
            self.parameters["g"],
            self.parameters["sigma"],
        )
        lvl = r_support[1]
        e = (r - r_support) / rg.sqrt(rg.norm_2(r - r_support))
        reaction = rg.if_else(
            lvl <= 0,
            e * rg.dot(e, m * g * rg.array([0, 1])) * lvl * sigma,
            rg.array([0.0, 0.0]),
        )
        return -reaction
