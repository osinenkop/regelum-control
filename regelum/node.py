from typing import List, Any, Dict, Optional, Tuple
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import singledispatch
from enum import Enum, auto

class Node(ABC):
    @dataclass
    class State:
        state_name: str
        dims: Tuple[int, ...]
    
    @dataclass
    class Inputs:
        inputs: List[str]
        def add_inputs(self, new_inputs: List[str]):
            self.inputs.extend(new_inputs)

    def __init__(self, inputs: Optional[List[str]] = None, state: Optional[State] = None) -> None:
        if inputs is not None:
            self.inputs = self.Inputs(inputs)
        else:
            self.inputs = self.Inputs([])

        if state is not None:
            self.state = state

    @abstractmethod
    def compute_state_dynamics(self, time, env_state, inputs): ...

    @abstractmethod
    def reset(self, env_state): ...

class Transistor:
    class Type(Enum):
        ode = auto()
        sde = auto()
        discrete = auto()


@singledispatch
def build_transistor(transistor_type, state_dynamics_function, state: Node.State, inputs: Node.Inputs):
    raise NotImplementedError
@build_transistor.register
def _(transistor_type: Transistor.Type.ode, state_dynamics_function, state: Node.State, inputs: Node.Inputs):
    print('This is an RK45 transistor')
@build_transistor.register
def _(transistor_type: Transistor.Type.sde, state_dynamics_function, state: Node.State, inputs: Node.Inputs):
    print('This is an SDE transistor')

@build_transistor.register
def _(transistor_type: Transistor.Type.discrete, state_dynamics_function, state: Node.State, inputs: Node.Inputs):
    print('This is a discrete time transistor')

class WhateverNode(Node):
    state = Node.State("whatever_node")
    inputs = Node.Inputs(["input1", "input2"])

    @Transistor.Type.ode
    def compute_state_dynamics(self, time, env_state, inputs): ...

node = WhateverNode(inputs=["input1", "input2"])

class GraphBuilder:
    def __init__(self, nodes: List[Node]): ...

    @staticmethod
    def resolve(nodes: List[Node]): ...

    @staticmethod
    def get_transitions(nodes: List[Node]) -> List[Transistor]: ...

    def step(self): ...


