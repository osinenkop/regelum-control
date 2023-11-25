"""Contains a generic interface for systems (environments) as well as concrete systems as realizations of the former.

Remarks:

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""
from __future__ import annotations
import numpy as np
import casadi as cs

import regelum
from abc import ABC, abstractmethod
from .utilis import rg
from typing import Optional, Union, List, Dict, Tuple
from typing_extensions import Self
from regelum.typing import RgArray


class SystemInterface(regelum.RegelumBase, ABC):
    """Abstract base class defining the interface for system environments.

    This interface sets the foundation for specific system implementations,
    providing properties for common attributes and abstract methods that
    must be implemented by subclasses to define system dynamics and observations.

    Attributes:
        _name: Name identifier for the system.
        _system_type: A string representing the type of the system (e.g., `"diff_eqn"`).
        _dim_state: The number of state variables of the system.
        _dim_inputs: The number of input variables, typically control inputs.
        _dim_observation: The number of observation variables.
        _parameters: A dictionary of system parameters.
        _observation_naming: A list of strings naming each observation dimension.
        _states_naming: A list of strings naming each state dimension.
        _inputs_naming: A list of strings naming each input dimension.
        _action_bounds: A list of pairs defining the lower and upper bounds for each action dimension.
    """

    _name: Optional[str] = None
    _system_type: Optional[str] = None
    _dim_state: Optional[int] = None
    _dim_inputs: Optional[int] = None
    _dim_observation: Optional[int] = None
    _parameters: Optional[Dict[str, float]] = {}
    _observation_naming: Optional[List[str]] = None
    _state_naming: Optional[List[str]] = None
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
    def dim_inputs(self) -> int:
        if self._dim_inputs is None:
            raise ValueError("system._dim_inputs should be set")
        return self._dim_inputs

    @property
    def parameters(self) -> dict:
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
    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute the derivative of the state with respect to time given the current state and inputs.

        This abstract method must be implemented by the inheriting system classes to define the specific system
        dynamics for the simulation. It provides the fundamental mathematical model that describes how the state
        of the system evolves over time in response to (e.g control or noisy) inputs.

        It is intended to be used internally by the `regelum.system.SystemInterface.compute_state_dynamics` method which provides a public interface,
        handling dimensionality enforcement and batch processing if necessary. The '_compute_state_dynamics' method
        ensures that the core dynamics calculations are encapsulated within each system's subclass, promoting a clear
        separation of the dynamics computation from any input preprocessing or other wrapping functionality.

        The implementation of this method in derived classes will typically involve evaluating the differential equations
        that govern the system's dynamics, returning the rate of change of the system's state variables.

        Args:
            time: The current simulation time.
            state: The current state vector of the system, where state[i] corresponds to the i-th state variable.
            inputs: The current input vector to the system, where inputs[i] corresponds to the i-th control input.

        Returns:
            The computed rate of change of the system's state vector.

        Note:
            This method does not update the state of the system. It only returns the derivative which may then
            be used by a numerical integrator to simulate the system's behavior over time.
        """
        pass

    def compute_state_dynamics(
        self,
        time: Union[float, cs.MX],
        state: RgArray,
        inputs: RgArray,
        _native_dim: bool = False,
    ) -> RgArray:
        """Computes the state dynamics of the system given the current state and inputs.

        This method is a public wrapper for the abstract method `_compute_state_dynamics`, which must be implemented by derived classes.

        Arguments:
            time: Current simulation time.
            state: The current state of the system.
            inputs: The current inputs to the system.
            _native_dim (optional): A flag indicating whether the method should maintain the original dimensionality of its inputs.

        Returns:
            The computed state dynamics.
        """
        if not _native_dim:
            return rg.force_row(
                self._compute_state_dynamics(
                    time, rg.force_column(state), rg.force_column(inputs)
                )
            )
        else:
            return self._compute_state_dynamics(time, state, inputs)

    @abstractmethod
    def _get_observation(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        pass

    def get_observation(
        self,
        time: Union[float, cs.MX],
        state: RgArray,
        inputs: RgArray,
        _native_dim: bool = False,
        is_batch: bool = False,
    ) -> RgArray:
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

    def apply_action_bounds(self, action: np.array) -> np.array:
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
    """Represents a composed system created from combining two subsystems.

    The composed system allows for creating complex system dynamics by
    combining simpler subsystems. The outputs of one subsystem can be
    connected to the inputs of another through an input/output mapping.

    Attributes are inherited and further defined based on the combination of subsystems.

    An instance of this class is being created automatically when applying a `@` operation on two systems.
    """

    def __init__(
        self,
        sys_left: Union[System, Self],
        sys_right: Union[System, Self],
        io_mapping: Optional[List[int]] = None,
        output_mode="right",
        state_naming: List[str] = None,
        inputs_naming: List[str] = None,
        observation_naming: List[str] = None,
        action_bounds: List[List[float]] = None,
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
    def __get_routing(io_mapping: List[int]):
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

    def _compute_state_dynamics(
        self,
        time: Union[float, cs.MX],
        state: RgArray,
        inputs: RgArray,
        _native_dim: bool = True,
    ) -> RgArray:
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

    def _get_observation(
        self,
        time: Union[float, cs.MX],
        state: RgArray,
        inputs: RgArray,
        _native_dim: bool = False,
    ) -> RgArray:
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

    def receive_action(self, action: np.array) -> None:
        self.inputs = action

    def update_system_parameters(self, inputs: Dict[str, float]) -> None:
        assert isinstance(inputs, dict)
        self.sys_left.update_system_parameters(inputs)
        self.sys_right.update_system_parameters(inputs)

    def permute_state(self, permutation: Union[List[int], np.array]) -> Self:
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

    def get_inverse_permutation(
        self, permutation: Union[List[int], np.array]
    ) -> np.array:
        self.current_permutation = permutation
        permutation = np.asanyarray(permutation)
        inverse_permutation = np.empty_like(permutation)
        inverse_permutation[permutation] = np.arange(permutation.size)
        return inverse_permutation

    def compose(
        self,
        sys_right: SystemInterface,
        io_mapping: Optional[List[int]] = None,
        output_mode: str = "state",
    ) -> ComposedSystem:
        return ComposedSystem(
            self, sys_right, io_mapping=io_mapping, output_mode=output_mode
        )

    def __matmul__(self, sys_right: SystemInterface) -> Self:
        return self.compose(sys_right)


class System(SystemInterface):
    """Class representing a controllable system with predefined dynamics and observations.

    Attributes and methods are inherited from SystemInterface. This class
    serves as a base implementation of a system with additional functionality
    to receive control inputs and update system parameters.
    """

    def __init__(
        self,
        system_parameters_init: Dict[str, float] = None,
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

    def _get_observation(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ):
        return state

    def receive_action(self, action: np.array) -> None:
        self.inputs = action

    def update_system_parameters(self, inputs: Dict[str, float]) -> Dict[str, float]:
        assert isinstance(inputs, dict)
        self._parameters.update(inputs)
        return self.parameters

    def compose(
        self,
        sys_right: SystemInterface,
        io_mapping: Optional[List[int]] = None,
        output_mode: str = "state",
    ):
        return ComposedSystem(
            self, sys_right, io_mapping=io_mapping, output_mode=output_mode
        )

    def __matmul__(self, sys_right: SystemInterface) -> Self:
        return self.compose(sys_right)


class KinematicPoint(System):
    """System representing a simple 2D kinematic point that can move in any direction."""

    _name = "kinematic-point"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _observation_naming = _state_naming = ["x", "y"]
    _inputs_naming = ["v_x", "v_y"]
    _action_bounds = [[-10.0, 10.0], [-10.0, 10.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        for i in range(rg.shape(inputs)[0]):
            Dstate[i] = inputs[i]

        return Dstate


class InvertedPendulum(System):
    """System representing an inverted pendulum, with state representing angle and angular velocity."""

    _name = "inverted-pendulum"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 1
    _dim_observation = 2
    _parameters = {"m": 1, "g": 9.8, "l": 1}
    _observation_naming = _state_naming = ["angle", "angular velocity"]
    _inputs_naming = ["momentum"]
    _action_bounds = [[-20.0, 20.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
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

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        Dstate = rg.zeros(self.dim_state, prototype=(state, inputs))

        Dstate[0] = inputs[0] * rg.cos(state[2])
        Dstate[1] = inputs[0] * rg.sin(state[2])
        Dstate[2] = inputs[1]

        return Dstate


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

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
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
    """System yielding Non-holonomic double integrator when composed with non-holomonic three-wheeled robot."""

    _name = "integral-parts"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _parameters = {"m": 10, "I": 1}

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, I = self.parameters["m"], self.parameters["I"]

        Dstate[0] = 1 / m * inputs[0]
        Dstate[1] = 1 / I * inputs[1]

        return Dstate


three_wheeled_robot_alternative = (Integrator() @ ThreeWheeledRobotNI()).permute_state(
    [3, 4, 0, 1, 2]
)


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

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
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

    def _get_observation(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ):
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

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
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


class LunarLander(System):
    """Lunar lander system.

    The basis of this system is taken from the [following paper](https://web.aeromech.usyd.edu.au/AMME3500/Course_documents/material/tutorials/Assignment%204%20Lunar%20Lander%20Solution.pdf).

    The LunarLander class simulates a lunar lander during its descent to the moon's surface. It models both the
    vertical and lateral dynamics as well as the lander's rotation.
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

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute the state dynamics of the lunar lander based on its physical model.

        This method calculates the derivative of the lander's state variables, considering the applied forces from
        thrusters and the gravitational forces. It checks for landing conditions to alter the dynamics accordingly, such
        as setting velocities to zero upon touching down.

        Args:
            time: The current simulation time.
            state: The current state of the lunar lander.
            inputs: The current inputs to the lander.

        Returns:
            The state vector's derivative with respect to time.
        """
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

        left_support, right_support = self._compute_supports_geometry(state[:2], theta)

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

    def _compute_pendulum_dynamics(
        self,
        angle: Union[RgArray, float],
        angle_dot: Union[RgArray, float],
        prototype: RgArray,
    ) -> RgArray:
        """Compute the dynamics of the lander when it is in contact with the ground, constrained like a pendulum.

        This method is used internally to modify the dynamics of the lander when one of its extremities touches down,
        causing it to behave like a pendulum.

        Args:
            angle: The angle of the pendulum.
            angle_dot: The derivative of the pendulum angle.
            prototype: A prototype array indicating the desired shape and type for the returned state derivative.

        Returns:
            The computed state vector's derivative with pendulum-like constraints applied.
        """
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

    def _compute_supports_geometry(
        self, xi: Union[RgArray, float], theta: Union[RgArray, float]
    ) -> Tuple[Union[RgArray, float], Union[RgArray, float]]:
        """Calculate the positions of the left and right supports of the lander.

        Args:
            xi: The position vector of the lander's center of gravity.
            theta: The current orientation angle of the lander.

        Returns:
            The positions of the left and right supports.
        """
        rotation_operator = rg.zeros((2, 2), prototype=xi)
        xi_2 = rg.zeros((2, 1), prototype=xi)
        xi_3 = rg.zeros((2, 1), prototype=xi)

        rotation_operator[0, 0] = rg.cos(theta)
        rotation_operator[0, 1] = -rg.sin(theta)
        rotation_operator[1, 0] = rg.sin(theta)
        rotation_operator[1, 1] = rg.cos(theta)

        a, r = self.parameters["a"], self.parameters["r"]
        xi_2[0, 0] = xi[0, 0] - a
        xi_2[1, 0] = xi[1, 0] - r
        xi_3[0, 0] = xi[0, 0] + a
        xi_3[1, 0] = xi[1, 0] - r

        xi_2_d = xi_2 - xi
        xi_3_d = xi_3 - xi

        xi_2_d_rot = rotation_operator @ xi_2_d
        xi_3_d_rot = rotation_operator @ xi_3_d
        xi_2_new = xi + xi_2_d_rot
        xi_3_new = xi + xi_3_d_rot
        return xi_2_new, xi_3_new

    def _compute_reaction(
        self,
        r: Union[RgArray, float],
        r_support: Union[RgArray, float],
    ) -> Union[RgArray, float]:
        """Compute the reaction forces when the lander's supports are in contact with the ground.

        Args:
            r: The position vector of the lander's center of gravity.
            r_support: The position vector of one of the lander's supports.

        Returns:
            np.array: The computed reaction forces at the point of contact.
        """
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

    def _get_observation(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        return inputs - rg.array(
            self.parameters["reference"], prototype=inputs, _force_numeric=True
        )

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        return inputs


class SystemWithConstantReference(ComposedSystem):
    """Composed system created by combining a system with a constant reference subtraction."""

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
        np_constant_reference = np.array(constant_reference).reshape(-1)
        super().__init__(
            sys_left=system,
            sys_right=constant_reference,
            io_mapping=None,
            output_mode="right",
            inputs_naming=system.inputs_naming,
            state_naming=system.state_naming,
            observation_naming=[
                s + f"-{np_constant_reference[i]}"
                for i, s in enumerate(system.state_naming)
            ],
            action_bounds=system.action_bounds,
        )


class LunarLanderReferenced(SystemWithConstantReference):
    def __init__(self):
        """Instantiate TwoTankReferenced."""
        super().__init__(
            system=LunarLander(), state_reference=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        )


class TwoTankReferenced(SystemWithConstantReference):
    def __init__(self):
        """Instantiate TwoTankReferenced."""
        super().__init__(system=TwoTank(), state_reference=[0.4, 0.4])
