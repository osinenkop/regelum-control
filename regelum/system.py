"""Contains a generic interface for systems (environments) as well as concrete systems as realizations of the former.

Note:
    The classes provided here do not perform any simulation mechanics itself.
    It contains only the update rule as defined in this section. The actual simulation routine is executed by a
    separate [`Simulator`][regelum.simulator] class, which leverages the update rule to create the corresponding integration scheme.
"""

from __future__ import annotations
import numpy as np
import casadi as cs
import torch
import regelum
from abc import ABC, abstractmethod

from . import callback
# from . import animation
from .utils import rg
from typing import Any, Optional, Union, List, Dict, Tuple
from typing_extensions import Self
from regelum.typing import RgArray


class SystemInterface(regelum.RegelumBase, ABC):
    """Abstract base class defining the interface for system environments.

    This interface sets the foundation for specific system implementations, providing properties for common attributes
    and abstract methods that must be implemented by subclasses to define system dynamics and observations.

    Attributes:
        _name: Name identifier for the system.
        _system_type: A string representing the type of the system (e.g., `"diff_eqn"`).
        _dim_state: The number of state variables of the system.
        _dim_inputs: The number of input variables, typically control inputs. In regelum we commonly use inputs parameter for actions.
        _dim_observation: The number of observation variables.
        _parameters: A dictionary of system parameters.
        _observation_naming: A list of strings naming each observation dimension.
        _states_naming: A list of strings naming each state dimension.
        _inputs_naming: A list of strings naming each input dimension.
        _action_bounds: A list of pairs defining the lower and upper bounds for each action dimension.
            Action bounds are defined as a list of [min, max] pairs for each action dimension
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
        dynamics for the simulation. It provides the fundamental mathematical model that describes how the state of the
        system evolves over time in response to (e.g control or noisy) inputs.

        It is intended to be used internally by the [`SystemInterface.compute_state_dynamics`][regelum.system.SystemInterface.compute_state_dynamics] method
        which provides a public interface, handling dimensionality enforcement and batch processing if necessary. The
        `_compute_state_dynamics` method ensures that the core dynamics calculations are encapsulated within each
        system's subclass, promoting a clear separation of the dynamics computation from any input preprocessing or
        other wrapping functionality.

        The implementation of this method in derived classes will typically involve evaluating the differential
        equations that govern the system's dynamics, returning the rate of change of the system's state variables.

        Args:
            time: The current simulation time.
            state: The current state vector of the system, where state[i] corresponds to the i-th state variable.
            inputs: The current input vector to the system, where inputs[i] corresponds to the i-th control input.

        Returns:
            The computed rate of change of the system's state vector.

        Note:
            This method does not update the state of the system. It only returns the derivative which may then be used
            by a numerical integrator to simulate the system's behavior over time.
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

        This method is a public wrapper for the abstract method `_compute_state_dynamics`, which must be implemented by
        derived classes.

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
        """Generate the observation of the system given the current state and inputs.

        This abstract method must be implemented by subclasses and is central to the observation-generating process of
        the system. In reinforcement learning and control systems, observations are the perceivable parts of the
        system's state and inputs that are accessible at a particular point in time. These observations are used by an
        agent or a controller to make decisions or by evaluation tools to assess the system's performance.

        The `_get_observation` method specifically handles the transformation of the system's state and inputs into a
        format best suited for decision-making or evaluation. This transformation may include, but is not limited to,
        sensor models, noise injection, feature extraction, or scaling.

        Args:
            time: The current simulation time or step.
            state: The current state vector of the system.
            inputs: The current input vector given to the system.

        Returns:
            An observation vector representing the perceived state of the system at the current time.

        Note:
            Implementing classes should define what makes up an observation in the context of their specific system.
            The method should aim to represent realistically perceivable quantities
        """
        pass

    def get_observation(
        self,
        time: Union[float, cs.MX],
        state: RgArray,
        inputs: RgArray,
        _native_dim: bool = False,
        is_batch: bool = False,
    ) -> RgArray:
        """Public method that wraps around the `_get_observation` abstract method to produce observations.

        This public interface method calls the internal abstract method
        [`_get_observation`][regelum.system.SystemInterface._get_observation] and applies additional
        formatting or dimensionality adjustments. It provides flexibility in handling observations on a per-instance or
        batch processing basis, which can be essential for efficiency in various simulations and algorithm
        implementations.

        Args:
            time: The current simulation time or step.
            state: The current state of the system.
            inputs: The current inputs given to the system.
            _native_dim: A flag indicating if the native dimensionality from `_get_observation` should be preserved.
                Defaults to False, which reshapes the output to align with expected dimensions in batch processing or
                other uses.
            is_batch: A flag indicating if the method should handle inputs and states in a batched form, processing multiple instances simultaneously.
                This is useful for vectorized operations that consider multiple simulation instances at once.

        Returns:
            Formatted observation vector(s) consistent with the system's defined observation space structure and
            sizing.
        """
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
            observations = rg.zeros(
                (state.shape[0], self.dim_observation), prototype=state
            )
            for i in range(state.shape[0]):
                observations[i, :] = self.get_observation(
                    time=time,
                    state=state[i, :],
                    inputs=inputs[i, :],
                    is_batch=False,
                )
            return observations

    def apply_action_bounds(self, action: np.array) -> np.array:
        """Apply the system's predefined action bounds to the given action vector.

        This method is responsible for ensuring that all actions executed by the system are within the specified safe
        or valid operating ranges. The enforcement of action bounds is a critical safety feature for many physical
        systems and a way to align virtual agent behavior with feasible actions in reinforcement learning environments.

        Args:
            action: A vector of actions proposed for the system, which may include elements outside the permissible bounds.

        Returns:
            A clipped action vector where each element is within the predefined action bounds of the system. If no
            bounds are defined, the input vector is returned unchanged.

        Note:
            The action bounds are typically set during the instantiation of a system class and represent the maximum
            and minimum values allowed for each action variable.
        """
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

    The composed system allows for creating complex system dynamics by combining simpler subsystems. The outputs of one
    subsystem can be connected to the inputs of another through an input/output mapping.

    Attributes are inherited and further defined based on the combination of subsystems.

    An instance of this class is being created automatically when applying a `@` operation on two systems.

    The constructor creates a complex system by interconnecting the outputs of one system (`sys_left`) to the inputs of
    another (`sys_right`) based on an optional input/output mapping. The combination can be thought of as a way of
    layering or cascading systems to create a more intricate overall system architecture.
    """

    def __call__(self) -> Any:
        """Needed for instantiation through _target_ in hydra configs

        Returns:
            Self
        """
        return self

    def __init__(
        self,
        sys_left: Union[System, Self],
        sys_right: Union[System, Self],
        io_mapping: Optional[List[int]] = None,
        output_mode: str = "right",
        state_naming: List[str] = None,
        inputs_naming: List[str] = None,
        observation_naming: List[str] = None,
        action_bounds: List[List[float]] = None,
    ):
        """Initialize a ComposedSystem by combining two subsystems.

        Args:
            sys_left: The first subsystem whose outputs can be connected to another system's inputs.
            sys_right: The second subsystem which can receive connections from the 'left' system.
            io_mapping: An optional list defining the connections between the 'left'
                system state components and the 'right' system input components. Each item represents an index from the
                left system whose output is to be fed as an input to the same index in the right system. Defaults to
                None, which implies a direct connection of all available states to inputs.
            output_mode: A string specifying how to aggregate and output the observations from the composed system.
                Options are `"state"`, `"right"`, or `"both"`. Defaults to "right" which returns the observation of the right
                system only.
            state_naming: Custom naming for the state variables of the ComposedSystem.
            inputs_naming: Custom naming for the input variables of the ComposedSystem.
            observation_naming: Custom naming for the observation variables of the ComposedSystem.
            action_bounds: Action bounds for the composite system,
                defined as a list of [min, max] pairs for each action dimension.

        Note:
            The `output_mode` parameter affects how the state and observation dimensions are defined in the
            ComposedSystem. The composition modifies the routing of signals between systems and, depending on the mode,
            changes the reported system observations.
        """
        self._state_naming = state_naming
        self._inputs_naming = inputs_naming
        self._observation_naming = observation_naming
        self._action_bounds = (
            sys_left.action_bounds if action_bounds is None else action_bounds
        )

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
    def __get_routing(io_mapping: List[int]) -> Tuple[np.array, np.array]:
        """Generate routing indices for state-to-input mapping of connected subsystems.

        This internal static method processes the input/output mapping provided during the composition of two subsystems.
        The method converts the mapping into index arrays that can be used to route the outputs of the left subsystem
        to the inputs of the right subsystem during the computation of state dynamics and observations.

        Args:
            io_mapping: A list defining how outputs (state) of the left subsystem are to be routed
                to inputs of the right subsystem. Each list item represents an index from the
                left system's output matched to an index in the right system's input.

        Returns:
            Two numpy arrays, where the first one contains indices of the left system's
                outputs, and the second one contains indices of the right system's inputs.
        """
        io_mapping_extended = []
        if len(io_mapping) > 0:
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
        else:
            rout_idx, occupied_idx = rg.array([]), rg.array([])
        return rout_idx, occupied_idx

    def _compute_state_dynamics(
        self,
        time: Union[float, cs.MX],
        state: RgArray,
        inputs: RgArray,
        _native_dim: bool = True,
    ) -> RgArray:
        """Compute the state dynamics of the composed system based on connected subsystems' dynamics.

        This method combines the individual state dynamics computations of the left and right subsystems that form the composed system.
        It appropriately routes outputs from the left subsystem as inputs to the right subsystem based on the established input/output mapping.
        State dynamics of each subsystem are then calculated individually, and the results are consolidated into a single state dynamics vector
        representing the dynamics of the entire composed system.

        Args:
            time: The current simulation time.
            state: The combined state vector of both subsystems, concatenated vertically (left subsystem states followed by right subsystem states).
            inputs: The input vector provided to the composed system.
            _native_dim: A boolean indicating whether to preserve the native dimensionality of the state and input vectors.

        Returns:
            A state dynamics vector for the whole composed system reflecting the rates of change of state variables.
        """
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
        if len(self.occupied_idx) > 0:
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
        final_dstate_vector = (
            rg.hstack((rg.force_row(dstate_of_left), rg.force_row(dstate_of_right)))
            if not isinstance(dstate_of_left, (np.ndarray, torch.Tensor))
            else rg.hstack((dstate_of_left, dstate_of_right))
        )

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
        """Compute the observation of the composed system based on the observations from connected subsystems.

        This method uses the internal observations generation of the left and right subsystems to produce the overall composed system's observation.
        Depending on the chosen output mode ('state', 'right', or 'both'), the method prepares the observation vector by either using the state of the
        composed system, the observation from the right subsystem only, or the concatenation of both subsystems' observations.

        Args:
            time: The current simulation time.
            state: The combined state vector for both subsystems.
            inputs: The input vector provided to the composed system.
            _native_dim: A boolean flag to indicate if the method should maintain the native dimensionality of its inputs.

        Returns:
            An observation vector as per the defined output mode of the composed system.
        """
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
        if len(self.occupied_idx) > 0:
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
        """Updates the parameters of both component subsystems based on a dictionary of new parameter values.

        This method propagates any system parameter updates to both the left and right subsystems that form the ComposedSystem.
        It is a convenient way to adjust or tune parameters affecting the dynamics and behavior of the overall system.

        Args:
            inputs: A dictionary where keys represent parameter names to update, and values represent the new parameter values.
            It is expected that both subsystems share a common parameter naming convention if the same parameters are to be updated across both systems.

        Note:
            Parameters that do not exist in either or both of the subsystems are ignored, and no exception is raised for such cases.
        """
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
        """Calculate the inverse of a given permutation array.

        This method is particularly useful when a specific output ordering of the composed system's state vector
        needs to be reversed to its original form, such as when unpacking state vectors for separate subsystem processing.

        Args:
            permutation (Union[List[int], np.array]): The permutation array for which the inverse is to be computed.

        Returns:
            A numpy array representing the inverse permutation.

        Note:
            The given permutation array must represent a valid permutation of its indices for the inverse to be calculated correctly.
        """
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
        """Create a new ComposedSystem by appending another system to the current one.

        This method allows for creating multi-layered system structures by successively combining systems.
        The right system specified will receive a connection from the current one as defined by the input/output mapping.

        Args:
            sys_right: The system to compose with the current one on the right.
            io_mapping: An optional list defining the state-to-input connections between the current system and the right system.
            output_mode: Determines how to create observations for the resulting composed system. Defaults to `"state"`.

        Returns:
            A new instance of ComposedSystem containing the current system as the 'left' system and the specified right system as the 'right' system,
                and connected according to the specified mapping and output mode.
        """
        return ComposedSystem(
            self, sys_right, io_mapping=io_mapping, output_mode=output_mode
        )

    def __matmul__(self, sys_right: SystemInterface) -> Self:
        """Special method to create a new ComposedSystem via the `@` operator.

        Using the `@` operator constructs a new ComposedSystem with the current instance acting as the left system and the parameter system becoming the right system in the composition.
        By default, the connections between systems are determined based on the indices of their states and inputs if an explicit mapping is not provided.

        Args:
            sys_right: The system to be connected to the current system.

        Returns:
            A new ComposedSystem instance representing the composition of the current and right systems.

        Example:
            ```python
            composed_system = system1 @ system2  # Compose system1 with system2.
            ```
        """
        return self.compose(sys_right)


# @animation.StateAnimation.attach
@callback.DefaultAnimation.attach
class System(SystemInterface):
    """Class representing a controllable system with predefined dynamics and observations.

    Attributes and methods are inherited from SystemInterface. This class serves as a base implementation of a system
    with additional functionality to receive control inputs and update system parameters.
    """

    def __init__(
        self,
        system_parameters_init: Dict[str, float] = None,
        state_init: Optional[np.ndarray] = None,
        inputs_init: Optional[np.ndarray] = None,
    ):
        """Initialize an instance of a system with optional overrides for system parameters, initial state, and inputs.

        Args:
            system_parameters_init: A dictionary containing system parameters
                that override the default parameters.

            state_init: An initial state to set for the system. This argument is
                currently not utilized in the system's logic and serves as a placeholder for
                potential future functionality where a system's state could be set at instantiation.

            inputs_init: Initial inputs to set for the system. Similar to
                state_init, this argument is not currently used in the implementation and is
                maintained for potential future enhancements where system inputs can be
                initialized upon creation.

        Note:
            The current implementation of the system class treats systems as static entities.
            The `state_init` and `inputs_init` parameters are part of the method signature
            for potential future use. As of now, they have no effect on the system's behavior
            and are not used in the code.
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
    ) -> RgArray:
        """Return state vector by default.

        In many cases the observation is simply the state vector, so it is quite reasonable to just
        set the observation to the state vector by default.

        Args:
            time: The current simulation time or step.
            state: The current state vector of the system.
            inputs: The current input vector given to the system.

        Returns:
            An observation vector representing the perceived state of the system at the current time.

        Note:
            Implementing classes should define what makes up an observation in the context of their specific system.
            The method should aim to represent realistically perceivable quantities
        """
        return state

    def receive_action(self, action: np.array) -> None:
        self.inputs = action

    def update_system_parameters(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Simply updates `_parameters` dict."""
        assert isinstance(inputs, dict)
        self._parameters.update(inputs)
        return self.parameters

    def compose(
        self,
        sys_right: SystemInterface,
        io_mapping: Optional[List[int]] = None,
        output_mode: str = "state",
    ) -> ComposedSystem:
        """Compose the current system with another system.

        Args:
            sys_right: The system to be composed with.
            io_mapping: A list specifying the input-output mapping between the current system and sys_right. Defaults to None.
            output_mode: The output mode of the composed system. Defaults to "state".

        Returns:
            The composed system.
        """
        return ComposedSystem(
            self, sys_right, io_mapping=io_mapping, output_mode=output_mode
        )

    def __matmul__(self, sys_right: SystemInterface) -> Self:
        """Special method to create a new ComposedSystem via the `@` operator.

        Using the `@` operator constructs a new ComposedSystem with the current instance acting as the left system and the parameter system becoming the right system in the composition.
        By default, the connections between systems are determined based on the indices of their states and inputs if an explicit mapping is not provided.

        Args:
            sys_right: The system to be connected to the current system.

        Returns:
            A new ComposedSystem instance representing the composition of the current and right systems.

        Example:
            composed_system = system1 @ system2  # Compose system1 with system2.
        """
        return self.compose(sys_right)

@callback.OmnirobotAnimation.attach
class KinematicPoint(System):
    """System representing a simple 2D kinematic point (see [here](../systems/kin_point.md) for details)."""

    _name = "kinematic-point"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _observation_naming = _state_naming = ["x [m]", "y [m]"]
    _inputs_naming = ["v_x [m/s]", "v_y [m/s]"]
    _action_bounds = [[-10.0, 10.0], [-10.0, 10.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute [right-hand side](../systems/kin_point.md#system-dynamics) of kinematic point.

        Args:
            time: Current time.
            state: Current state.
            inputs: Current control inputs (i. e. action).

        Returns:
            Right-hand side of the kinematic point.
        """

        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        for i in range(rg.shape(inputs)[0]):
            Dstate[i] = inputs[i]

        return Dstate

@callback.PendulumAnimation.attach
class InvertedPendulum(System):
    """System representing an [inverted pendulum](../systems/inv_pendulum.md), with state representing angle and angular velocity."""

    _name = "inverted-pendulum"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 1
    _dim_observation = 2
    _parameters = {"m": 1, "g": 9.8, "l": 1}
    _observation_naming = _state_naming = ["angle [rad]", "angular velocity [rad/s]"]
    _inputs_naming = ["momentum [kg*m/s]"]
    _action_bounds = [[-20.0, 20.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute [right-hand side](../systems/inv_pendulum.md#system-dynamics) of the inverted pendulum."""

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

@callback.ThreeWheeledRobotAnimation.attach
class ThreeWheeledRobotKinematic(System):
    r"""Implements the [kinematic three-wheeled robot](../systems/3wrobot_kin.md) (a.k.a. Brockett integrator)."""

    _name = "three-wheeled-robot-ni"
    _system_type = "diff_eqn"
    _dim_state = 3
    _dim_inputs = 2
    _dim_observation = 3
    _observation_naming = _state_naming = ["x [m]", "y [m]", "angle [rad]"]
    _inputs_naming = ["velocity [m/s]", "angular velocity [rad/s]"]
    _action_bounds = [[-25.0, 25.0], [-5.0, 5.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute [right-hand side](../systems/3wrobot_kin.md#system-dynamics) of the dynamic system.

        Args:
            time: Current time.
            state: Current state.
            inputs: Current control inputs (i. e. action).

        Returns:
            Right-hand side of the non-holonomic robot.
        """
        Dstate = rg.zeros(self.dim_state, prototype=(state, inputs))

        Dstate[0] = inputs[0] * rg.cos(state[2])
        Dstate[1] = inputs[0] * rg.sin(state[2])
        Dstate[2] = inputs[1]

        return Dstate

@callback.ThreeWheeledRobotAnimation.attach
class ThreeWheeledRobotDynamic(System):
    r"""Implements [dynamic three-wheeled robot](../systems/3wrobot_dyn.md)."""

    _name = "three-wheeled-robot"
    _system_type = "diff_eqn"
    _dim_state = 5
    _dim_inputs = 2
    _dim_observation = 5
    _parameters = {"m": 10, "I": 1}
    _observation_naming = _state_naming = [
        "x [m]",
        "y [m]",
        "angle [rad]",
        "l_velocity [m/s]",
        "angular_velocity [rad/s]",
    ]
    _inputs_naming = ["Force [kg*m/s^2]", "Momentum [kg*m/s]"]
    _action_bounds = [[-50.0, 50.0], [-10.0, 10.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute [right-hand side](../systems/3wrobot_dyn.md#system-dynamics) of the dynamic system.

        Args:
            time: Current time.
            state: Current state.
            inputs: Current action.

        Returns:
            Right-hand side.
        """
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
    """System yielding non-holonomic double integrator when composed with [kinematic three-wheeled robot][regelum.system.ThreeWheeledRobotKinematic].

    See [here](../tutorials/systems.md#example-3-combining-integrator-with-kinematic-robot) for details.
    """

    _name = "integral-parts"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _parameters = {"m": 10, "I": 1}
    _action_bounds = [[-50.0, 50.0], [-10.0, 10.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute [right-hand side](../tutorials/systems.md#example-3-combining-integrator-with-kinematic-robot) of the dynamic system.

        Args:
            time: Current time.
            state: Current state.
            inputs: Current action.

        Returns:
            Right-hand side.
        """
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, I = self.parameters["m"], self.parameters["I"]

        Dstate[0] = 1 / m * inputs[0]
        Dstate[1] = 1 / I * inputs[1]

        return Dstate

@callback.CartpoleAnimation.attach
class CartPole(System):
    """[Cart pole system](../systems/cartpole.md) without friction."""

    _name = "cartpole"
    _system_type = "diff_eqn"
    _dim_state = 4
    _dim_inputs = 1
    _dim_observation = 5
    _parameters = {"m_c": 0.1, "m_p": 2.0, "g": 9.81, "l": 0.5}
    _observation_naming = _state_naming = [
        "angle [rad]",
        "x [m]",
        "angle_dot [rad/s]",
        "x_dot [m/s]",
    ]
    _inputs_naming = ["force [kg*m/s^2]"]
    _action_bounds = [[-50.0, 50.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute [right-hand side](../systems/cartpole.md#system-dynamics) of the dynamic system.

        Args:
            time: Current time.
            state: Current state.
            inputs: Current action.

        Returns:
            Right-hand side.
        """
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
        Force = inputs[0]
        sin_theta = rg.sin(theta)
        cos_theta = rg.cos(theta)
        theta_acc = (
            (
                g * sin_theta
                - cos_theta * (Force + m_p * l * theta_dot**2 * sin_theta) / (m_c + m_p)
            )
            / l
            / (4 / 3 - m_p * (cos_theta**2) / (m_c + m_p))
        )

        Dstate[0] = theta_dot
        Dstate[1] = x_dot
        Dstate[2] = theta_acc
        Dstate[3] = (
            Force + m_p * l * (theta_dot**2 * sin_theta - theta_acc * cos_theta)
        ) / (m_c + m_p)

        return Dstate

    def _get_observation(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ):
        theta = state[0]
        x = state[1]
        theta_dot = state[2]
        x_dot = state[3]

        # theta_observed = theta - rg.floor(theta / (2 * np.pi)) * 2 * np.pi
        # theta_observed = rg.if_else(
        #     theta_observed > np.pi,
        #     theta_observed - 2 * np.pi,
        #     theta - rg.floor(theta / (2 * np.pi)) * 2 * np.pi,
        # )
        # return rg.hstack([theta_observed, x, theta_dot, x_dot])
        # return rg.hstack([rg.sin(theta), 1 - rg.cos(theta), theta_dot, x_dot])
        # return rg.hstack([rg.sin(theta), 1 - rg.cos(theta), x, theta_dot, x_dot])
        return rg.hstack([rg.sin(theta), 1 - rg.cos(theta), x, theta_dot, x_dot])


class CartPolePG(CartPole):
    _name = "cartpole_pg"
    _system_type = "diff_eqn"
    _dim_state = 4
    _dim_inputs = 1
    _dim_observation = 5
    _parameters = {"m_c": 0.1, "m_p": 2.0, "g": 9.81, "l": 0.5}
    _observation_naming = _state_naming = [
        "angle [rad]",
        "x [m]",
        "angle_dot [rad/s]",
        "x_dot [m/s]",
    ]
    _inputs_naming = ["force [kg*m/s^2]"]
    _action_bounds = [[-50.0, 50.0]]
    _dim_observation = 4

    def _get_observation(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ):
        theta = state[0]
        x = state[1]
        theta_dot = state[2]
        x_dot = state[3]
        # theta_observed = theta - rg.floor(theta / (2 * np.pi)) * 2 * np.pi
        # theta_observed = rg.if_else(
        #     theta_observed > np.pi,
        #     theta_observed - 2 * np.pi,
        #     theta - rg.floor(theta / (2 * np.pi)) * 2 * np.pi,
        # )
        # return rg.hstack([theta_observed, x, theta_dot, x_dot])
        return rg.hstack([rg.sin(theta), 1 - rg.cos(theta), theta_dot, x_dot])
        # return rg.hstack([rg.sin(theta), 1 - rg.cos(theta), x, theta_dot, x_dot])


@callback.TwoTankAnimation.attach
class TwoTank(System):
    """This module simulates a [Two-Tank System](../systems/2tank.md).

    Notes:
        Please be aware that, despite the presence of a link to the [tutorial](../systems/2tank.md),
        this current system implementation does not replicate the tutorial's version.
        The primary distinction lies in the handling of observations: in this variant,
        observations are equivalent to the system's state variables.

        For an alternative version where the observation is differentiated from the state by a
        reference value (specifically, $(0.4, 0.4)$ subtracted from the state), please consult
        [`TwoTankReferenced`][regelum.system.TwoTankReferenced].
        It is this [`TwoTankReferenced`](regelum.system.TwoTankReferenced) model
        that faithfully reproduces the system outlined in the [tutorial](../systems/2tank.md).
    """

    _name = "two-tank"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 1
    _dim_observation = 2
    _parameters = {"tau1": 18.4, "tau2": 24.4, "K1": 1.3, "K2": 1.0, "K3": 0.2}
    _observation_naming = _state_naming = [
        "Intake Level [m]",
        "Sink Level [m]",
    ]
    _inputs_naming = ["Pressure [Pa]"]
    _action_bounds = [[0.0, 1.0]]

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """Compute [right-hand side](../systems/2tank.md#system-dynamics) of the dynamic system.

        Notes:
            Please be aware that, despite the presence of a link to the [tutorial](../systems/2tank.md),
            this current system implementation does not replicate the tutorial's version.
            The primary distinction lies in the handling of observations: in this variant,
            observations are equivalent to the system's state variables.

            For an alternative version where the observation is differentiated from the state by a
            reference value (specifically, $(0.4, 0.4)$ subtracted from the state), please consult
            [`TwoTankReferenced`][regelum.system.TwoTankReferenced].
            It is this [`TwoTankReferenced`](regelum.system.TwoTankReferenced) model
            that faithfully reproduces the system outlined in the [tutorial](../systems/2tank.md).

        Args:
            time: Current time.
            state: Current state.
            inputs: Current action.

        Returns:
            Right-hand side.
        """
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

@callback.LunarLanderAnimation.attach
class LunarLander(System):
    """[Lunar lander system](../systems/lunar_lander.md).

    The basis of this system is taken from the [following
    paper](https://web.aeromech.usyd.edu.au/AMME3500/Course_documents/material/tutorials/Assignment%204%20Lunar%20Lander%20Solution.pdf).

    The LunarLander class simulates a lunar lander during its descent to the moon's surface. It models both the
    vertical and lateral dynamics as well as the lander's rotation.

    Notes:
        Please be aware that, despite the presence of a link to the [tutorial](../systems/lunar_lander.md),
        this current system implementation does not replicate the tutorial's version.
        The primary distinction lies in the handling of observations: in this variant,
        observations are equivalent to the system's state variables.

        For an alternative version where the observation is differentiated from the state by a
        reference value (specifically, $(0.4, 0.4)$ subtracted from the state), please consult
        [`LunarLanderReferenced`][regelum.system.LunarLanderReferenced].
        It is this [`LunarLanderReferenced`](regelum.system.LunarLanderReferenced) model
        that faithfully reproduces the system outlined in the [tutorial](../systems/lunar_lander.md).
    """

    _name = "lander"
    _system_type = "diff_eqn"
    _dim_state = 6
    _dim_inputs = 2
    _dim_observation = 6
    _parameters = {"m": 10, "J": 3.0, "g": 1.625, "a": 1.0, "r": 1.0, "sigma": 1.0}
    _observation_naming = _state_naming = [
        "x [m]",
        "y [m]",
        "theta [rad]",
        "x_dot [m/s]",
        "y_dot [m/s]",
        "theta_dot [rad/s]",
    ]
    _inputs_naming = ["vertical force [kg*m/s^2]", "side force [kg*m/s^2]"]
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
        thrusters and the gravitational forces. It checks for landing conditions to alter the dynamics accordingly,
        such as setting velocities to zero upon touching down.

        Notes:
            Please be aware that, despite the presence of a link to the [tutorial](../systems/lunar_lander.md),
            this current system implementation does not replicate the tutorial's version.
            The primary distinction lies in the handling of observations: in this variant,
            observations are equivalent to the system's state variables.

            For an alternative version where the observation is differentiated from the state by a
            reference value (specifically, $(0.4, 0.4)$ subtracted from the state), please consult
            [`LunarLanderReferenced`][regelum.system.LunarLanderReferenced].
            It is this [`LunarLanderReferenced`](regelum.system.LunarLanderReferenced) model
            that faithfully reproduces the system outlined in the [tutorial](../systems/lunar_lander.md).

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
        """Compute the [dynamics](../systems/lunar_lander.md#system-dynamics) of the lander when it is in contact with the ground, constrained like a pendulum.

        This method is used internally to modify the dynamics of the lander when one of its extremities touches down,
        causing it to behave like a pendulum.

        Notes:
            Please be aware that, despite the presence of a link to the [tutorial](../systems/lunar_lander.md),
            this current system implementation does not replicate the tutorial's version.
            The primary distinction lies in the handling of observations: in this variant,
            observations are equivalent to the system's state variables.

            For an alternative version where the observation is differentiated from the state by a
            reference value (specifically, $(0.4, 0.4)$ subtracted from the state), please consult
            [`LunarLanderReferenced`][regelum.system.LunarLanderReferenced].
            It is this [`LunarLanderReferenced`](regelum.system.LunarLanderReferenced) model
            that faithfully reproduces the system outlined in the [tutorial](../systems/lunar_lander.md).

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


@callback.detach
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
                reference to be substracted from inputs, defaults to None
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
        np.array(constant_reference).reshape(-1)
        super().__init__(
            sys_left=system,
            sys_right=constant_reference,
            io_mapping=None,
            output_mode="right",
            inputs_naming=system.inputs_naming,
            state_naming=system.state_naming,
            observation_naming=[s + "-ref" for i, s in enumerate(system.state_naming)],
            action_bounds=system.action_bounds,
        )


class LunarLanderReferenced(SystemWithConstantReference):
    """Lunar lander referenced system.

    This class represents a tailored version of the [`LunarLander`](regelum.system.LunarLander)
    with specific enhancements to match the instructional framework provided by the [tutorial](../systems/lunar_lander.md).

    Notes:
        The [`LunarLanderReferenced`][regelum.system.LunarLanderReferenced] system varies from the original
        [`LunarLander`][regelum.system.LunarLander] in one significant aspect: the computation of the observation.
        In [`LunarLander`][regelum.system.LunarLander], the observation directly equals the current state.
        However, in [`LunarLanderReferenced`][regelum.system.LunarLanderReferenced],
        the observation is calculated by subtracting a fixed reference value of $(0, 1, 0, 0, 0, 0)$ from the current state,
        resulting in an adjusted observation value. This modification
        is designed for scenarios where the observation needs to be anchored to a particular reference point.

        Despite this alteration in the observation representation, the core dynamics of the state remain unaltered.
        They follow the same principles as elucidated in the [system dynamics section of the tutorial](../systems/lunar_lander.md#system-dynamics)
        and are consistent with the [`LunarLander`][regelum.system.LunarLander] class.
        This ensures that while the observation method has been modified, the fundamental
        behavior of the system **state** dynamics stays true to the original model.
    """

    def __init__(self):
        """Instantiate TwoTankReferenced."""
        super().__init__(
            system=LunarLander(), state_reference=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        )


class TwoTankReferenced(SystemWithConstantReference):
    """Two tank referenced system.

    This class represents a tailored version of the [`TwoTank`][regelum.system.TwoTank]
    with specific enhancements to match the instructional framework provided by the [tutorial](../systems/2tank.md).

    Notes:
        The [`TwoTankReferenced`][regelum.system.TwoTankReferenced] system varies from the original
        [`TwoTank`][regelum.system.TwoTank] in one significant aspect: the computation of the observation.
        In [`TwoTank`][regelum.system.TwoTank], the observation directly equals the current state.
        However, in [`TwoTankReferenced`][regelum.system.TwoTankReferenced],
        the observation is calculated by subtracting a fixed reference value of $(0.4, 0.4)$ from the current state,
        resulting in an adjusted observation value. This modification
        is designed for scenarios where the observation needs to be anchored to a particular reference point.

        Despite this alteration in the observation representation, the core dynamics of the state remain unaltered.
        They follow the same principles as elucidated in the [system dynamics section of the tutorial](../systems/2tank.md#system-dynamics)
        and are consistent with the [`TwoTank`][regelum.system.TwoTank] class.
        This ensures that while the observation method has been modified, the fundamental
        behavior of the system **state** dynamics stays true to the original model.

    """

    def __init__(self):
        """Instantiate TwoTankReferenced."""
        super().__init__(system=TwoTank(), state_reference=[0.4, 0.4])
