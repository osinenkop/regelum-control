"""This module contains the base class for all transistors that compute transitions of nodes composing an environment."""

from __future__ import annotations
from typing import Callable, Optional, Union

from regelum.environment.commutator import Commutator
from ..typing import TransitionMapType, StateType, StateType
from .. import rg
from abc import abstractmethod, ABC
from .node import Node, ComposedNode, State, Inputs
from operator import itemgetter


def call_once_only(func):
    def wrapper(*args, **kwargs):
        if not wrapper.called:
            wrapper.called = True
            return func(*args, **kwargs)

    wrapper.called = False
    return wrapper


class Transistor(ABC):
    """An entity representing a state transition of a node in the environment."""

    def __init__(
        self,
        step_size: float,
        time_final: Optional[float] = None,
    ) -> None:
        """Instantiate a bre-builded Transistor that only contains a generic state-transition function.

        The only method that should be implemented by the user is the 'create_transition_map' method.
        """
        self._transition_map: TransitionMapType
        self.step_size = step_size
        self.time_final = time_final

    @abstractmethod
    def create_transition_map(self) -> TransitionMapType: ...

    def transition_map(
        self,
        state: StateType,
        inputs: StateType,
    ) -> Union[StateType, StateType]:
        assert (
            self._transition_map is not None
        ), f"Transistor {self.__class__.__name__} is not initialized. Call 'build' method first."
        return (
            self._transition_map(state, inputs)
            if len(self.inputs.inputs.keys() - inputs.keys()) == 0
            else self.state_init
        )

    def step(self, commutator: Commutator) -> None:

        got_state = {k: v for k, v in commutator.items() if k in self.state.name}
        state = got_state if got_state is not None else self.state_init

        assert (
            state is not None
        ), "State value is not defined. Try to specify initial state of the node."
        new_state = self.transition_map(
            state=state,
            inputs={k: v for k, v in commutator.items() if k in self.inputs.inputs},
        )
        print(new_state)
        if isinstance(new_state, dict):
            commutator.update(**new_state)
        else:
            assert isinstance(
                self.state.name, str
            ), "State name must be a string for non-composed nodes."
            commutator[self.state.name] = new_state

    # @call_once_only
    def build(self, node: Union[Node, ComposedNode]) -> Transistor:
        self.node = node
        self.state = node.state
        self.inputs = node.inputs
        self.state_init = node.state_init
        self.get_state = (
            itemgetter(*node.state.name.split("+"))
            if isinstance(node, ComposedNode)
            else itemgetter(node.state.name)
        )
        self.state_dynamics_function = node.compute_state_dynamics
        self._transition_map = self.create_transition_map()

        return self


class DiscreteTransistor(Transistor):
    """A base class for computation of the evolution of the state of a node represented as a discrete state."""

    def create_transition_map(self):
        return self.state_dynamics_function


class ODETransistor(Transistor):
    """A base class for computation of the evolution of the state of a node represented as an ODE."""

    class IntegratorInterface(ABC):
        """An interface for an ODE integrator."""

        def __init__(
            self,
            state: State,
            inputs: Inputs,
            step_size: float,
            state_dynamics_function: Callable,
        ) -> None:
            """Instantiate an ODE integrator."""
            self.time = 0.0
            self.state_info = state
            self.inputs_info = inputs
            self.step_size = step_size
            self.state_dynamics_function = state_dynamics_function

        @abstractmethod
        def create(self): ...

    def __init__(
        self,
        time_final: float,
        step_size: float,
        time_start: float = 0.0,
    ) -> None:
        """Instantiate an ODETransistor."""
        super().__init__(
            step_size=step_size,
            time_final=time_final,
        )
        self.time = self.time_start = time_start


class CasADi(ODETransistor):
    """An ODETransistor that uses CasADi to compute the state transitions.

    Only supports action as an input parameter.
    """

    class CasADiIntegrator(ODETransistor.IntegratorInterface):
        """An ODE integrator that uses CasADi to compute the state transitions."""

        def create(self):
            from casadi import integrator

            state_symbolic = self.state_info.to_casadi_symbolic()
            inputs_symbolic_dict = self.inputs_info.to_casadi_symbolic_dict()
            inputs_symbolic_vector = (
                rg.vstack([rg.vec(k) for k in inputs_symbolic_dict.values()])
                if inputs_symbolic_dict
                else rg.DM([])
            )
            state_dynamics = self.state_dynamics_function(
                state_symbolic, inputs_symbolic_dict
            )

            DAE = {
                "x": state_symbolic,
                "p": inputs_symbolic_vector,
                "ode": state_dynamics,
            }

            return integrator("intg", "rk", DAE, 0, self.step_size)

    def create_transition_map(self):
        if isinstance(self.state, State):
            self.integrator = self.CasADiIntegrator(
                state=self.state,
                inputs=self.inputs,
                step_size=self.step_size,
                state_dynamics_function=self.state_dynamics_function,
            ).create()
            get_inputs = itemgetter(*self.inputs.inputs.keys())
            return lambda state, inputs: self.integrator(
                x0=state, p=rg.vstack([rg.vec(i) for i in get_inputs(inputs)])
            )["xf"].full()

        else:
            raise NotImplementedError("CasADi is not implemented for ComposedNodes")
