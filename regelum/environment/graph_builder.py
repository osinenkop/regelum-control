"""This module contains the graph builder class that builds and resolves the graph of nodes in the environment."""

from typing import List, Dict, Union
from regelum.environment.node import Node


class GraphBuilder:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.node_transistor_map = {node.state.name: node.transistor for node in nodes}
        self.node_state_inputs_map = {
            node.__class__.__name__: {
                "state": node.state.name,
                "inputs": node.inputs.names,
            }
            for node in nodes
        }

        self.ordered_nodes = self.resolve(self.node_state_inputs_map)
        self.fundamental_dt = min([node.transistor.step_size for node in self.nodes])

    def step(self): ...
