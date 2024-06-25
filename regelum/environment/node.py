"""This module contains the base class for all nodes in the environment."""

from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union, List
from abc import abstractmethod, ABC
from dataclasses import dataclass
from casadi import MX
import numpy as np

if TYPE_CHECKING:
    from .transistor import Transistor, DiscreteTransistor
from ..typing import (
    LeafStateType,
    RgArray,
    InputsDeclarationType,
    StateType,
    StateInitType,
)
from .. import rg
from .commutator import Commutator


@dataclass
class State:
    """A wrapper class for the state of a node in the environment."""

    name: str
    dims: Optional[Tuple[int, ...]] = None
    init: StateInitType

    def to_casadi_symbolic(self) -> MX:
        return rg.array_symb(self.dims, literal=self.name)

    def to_inputs(self) -> Inputs:
        return Inputs({self.name: self.dims})


@dataclass
class Inputs:
    """A wrapper class for the inputs of a node in the environment."""

    inputs: InputsDeclarationType

    @property
    def names(self):
        return list(self.inputs.keys())

    @property
    def is_empty(self):
        return len(self.inputs) == 0

    def to_casadi_symbolic(self):
        return (
            rg.vstack(
                [
                    rg.vec(rg.array_symb(dims, literal=k))
                    for k, dims in self.inputs.items()
                ]
            )
            if self.inputs is not None
            else rg.DM([])
        )

    def to_casadi_symbolic_dict(self):
        return {
            k: rg.vec(rg.array_symb(dims, literal=k)) for k, dims in self.inputs.items()
        }

    def extend(self, new_inputs: Inputs) -> None:
        self.inputs.update(new_inputs.inputs)


class Node(ABC):
    """An entity representing an atomic unit with time-dependent state.

    De-facto, a node is just a declaration of an evolving object and therefore is non-mutable.
    All time-dependent components of a node are stored in the `commutator` variable which is
    globally accessible to all nodes it was connected to.
    """

    def __init__(
        self,
        state_init: StateInitType,
        transistor: Transistor,
        inputs: Optional[Union[InputsDeclarationType, Inputs]] = None,
        state: Optional[Union[InputsDeclarationType, State]] = None,
        is_root: bool = False,
    ) -> None:
        """Instantiate a Node object with automatically built respective Transistor object."""
        if not hasattr(self, "inputs"):
            if inputs is not None:
                self.inputs = (
                    Inputs(inputs) if not isinstance(inputs, Inputs) else inputs
                )
            else:
                self.inputs = Inputs(dict())

        if not hasattr(self, "state"):
            if state is not None:
                self.state = (
                    State(name=list(state.keys())[0], dims=list(state.values())[0])
                    if isinstance(state, Dict)
                    else state
                )
            else:
                raise ValueError("State must be fully specified.")

        if callable(state_init):
            state_init = state_init()

        self.state_init: StateType = (
            {self.state.name: state_init}
            if not isinstance(state_init, dict)
            else state_init
        )
        self.is_root = is_root
        self.transistor = transistor.build(self)

    def connect(self, node: Node) -> Inputs:
        """Connect this node to another node by including the state of the other node to the inputs."""
        self.inputs.extend(node.state.to_inputs())
        return self.inputs

    @abstractmethod
    def compute_state_dynamics(
        self, state: StateType, inputs: StateType
    ) -> StateType: ...


class ComposedNode(Node):
    """A base class for nodes that are composed of other nodes."""

    class ComposedState:
        """A wrapper class for the state of a composed node in the environment."""

        def __init__(self, nodes: List[Union[Node, ComposedNode]]):
            """Instantiate a ComposedState object representing the state of a composed node."""
            self.name = "+".join([n.state.name for n in nodes])

        def __eq__(self, value: str) -> bool:
            return value in self.name.split("+")

    def __init__(self, nodes: List[Node], is_root: bool = False) -> None:
        """Instantiate a ComposedNode object."""
        self.nodes = nodes
        self.state = self.ComposedState(self.nodes)
        self.ordered_nodes = self.resolve(self.nodes)
        self.state_init = {
            state_name: state_init
            for node in self.ordered_nodes
            for state_name, state_init in node.state_init.items()
        }
        self.inputs = self.ordered_nodes[0].inputs
        from .transistor import DiscreteTransistor

        super().__init__(
            transistor=DiscreteTransistor(
                step_size=1.0,
                time_final=min(
                    (
                        node.transistor.time_final
                        if node.transistor.time_final is not None
                        else float("inf")
                    )
                    for node in nodes
                ),
            ),
            is_root=is_root,
            state_init=self.state_init,
        )

    @staticmethod
    def resolve(nodes: List[Union[Node, ComposedNode]]) -> List[Node]:
        """Resolves the order of nodes in the graph so that every node is executed only if all of its inputs are available as states of previously executed nodes."""
        node_state_inputs_map = {
            node.state.name: {
                "state": node.state,
                "inputs": node.inputs.names,
                "is_root": node.is_root,
                "state_init": node.state_init,
            }
            for node in nodes
        }
        assert len(set(node_state_inputs_map.keys())) == len(
            node_state_inputs_map
        ), "Duplicate node states detected"

        ordered_node_names: List[str] = []
        n_times_max = len(node_state_inputs_map)
        n_times_elapsed = 0
        while len(ordered_node_names) < len(node_state_inputs_map):
            assert n_times_elapsed < n_times_max, (
                "Graph cannot be resolved. Nodes not resolved "
                f"after {n_times_elapsed} interatons "
                f"are: {node_state_inputs_map.keys() - set(ordered_node_names)}."
            )
            for node_name, node_info in node_state_inputs_map.items():
                ordered_nodes_states: List[Union[State, ComposedNode.ComposedState]] = [
                    node_state_inputs_map[n]["state"] for n in ordered_node_names
                ]
                if node_name not in ordered_node_names:
                    if all(
                        input_name in ordered_nodes_states
                        for input_name in node_info["inputs"]
                    ) or (
                        node_info["is_root"] and (node_info["state_init"] is not None)
                    ):
                        ordered_node_names.append(node_name)
            n_times_elapsed += 1

        ordered_nodes: List[Union[Node, ComposedNode]] = []
        while len(ordered_nodes) < len(ordered_node_names):
            for node in nodes:
                if node.state.name in ordered_node_names:
                    ordered_nodes.append(node)
                    ordered_node_names.remove(node.state.name)

        return ordered_nodes

    def compute_state_dynamics(self, state: StateType, inputs: StateType) -> StateType:

        for node in self.ordered_nodes:
            if isinstance(node, ComposedNode):
                node.compute_state_dynamics(commutator)
            else:
                node.transistor.step(commutator)


class Clock(Node):
    """A node representing a clock with a fixed time step size."""

    state = State("Clock", (1,))

    def __init__(self, nodes: List[Node], time_start: float = 0.0) -> None:
        """Instantiate a Clock node with a fixed time step size."""
        from .transistor import DiscreteTransistor

        step_sizes = [node.transistor.step_size for node in nodes]
        if len(set(step_sizes)) > 1:
            self.fundamental_step_size = np.gcd.reduce(step_sizes)
        else:
            self.fundamental_step_size = step_sizes[0]
        super().__init__(
            transistor=DiscreteTransistor(
                step_size=self.fundamental_step_size,
                time_final=float("inf"),
            ),
            state_init=time_start,
        )

    def compute_state_dynamics(
        self, state: RgArray, inputs: Dict[str, RgArray]
    ) -> RgArray:
        return state[0] + self.fundamental_step_size


class Terminate(Node):
    """A node representing a termination condition."""

    postfix = "_terminate"
    inputs = Inputs({"Clock": (1,)})

    def __init__(self, node_to_terminate: Node) -> None:
        """Instantiate a Terminate node."""
        self.node_to_terminate = node_to_terminate
        self.state = State(node_to_terminate.state.name + self.postfix, (1,))
        self.connect(node_to_terminate)

        from regelum.environment.transistor import DiscreteTransistor

        super().__init__(
            transistor=DiscreteTransistor(
                step_size=node_to_terminate.transistor.step_size,
                time_final=node_to_terminate.transistor.time_final,
            ),
            state=self.state,
            inputs=self.inputs,
            state_init={self.state.name: False},
        )

    def compute_state_dynamics(
        self, state: RgArray, inputs: Dict[str, RgArray]
    ) -> bool:
        print(self.node_to_terminate)
        if self.node_to_terminate.transistor.time_final is not None:
            return (
                False
                if inputs["Clock"] < self.node_to_terminate.transistor.time_final
                else True
            )

        return False


# class Agent(Node):
#     """A node representing an agent producing actions passing into the plant."""

#     def __init__(self, plant_state_node: Node, step_size: float) -> None:
#         """Instantiate an Agent node."""
#         assert (
#             "action" in plant_state_node.inputs.inputs
#         ), f"Couldn't find action input in node {plant_state_node.__class__.__name__}"
#         self.state = State(name="action", dims=plant_state_node.inputs.inputs["action"])
#         self.connect(plant_state_node)

#         super().__init__(
#             transistor=DiscreteTransistor(
#                 step_size=step_size,
#                 time_final=plant_state_node.transistor.time_final,
#             ),
#             state=self.state,
#             inputs=self.inputs,
#         )

"""
node1 = Node()
node2 = Node()
node3 = Node()
node4 = Node()
node5 = Node()

nodes = [node1, node2, node3, node4, node5]

with GraphBuilder(nodes) as graph:
    graph.state_flow(node1 => node2 => node3).loop(n_interations=..., on_iteration=...).resolve()
"""
