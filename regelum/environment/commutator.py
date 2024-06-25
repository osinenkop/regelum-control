from ..typing import StateType, LeafStateType
from typing import Dict, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .transistor import Transistor
    from .node import Node


class Commutator:
    def __init__(self, state_tree: StateType) -> None:
        """Instantiate a Commutator that represents the state of a graph composed of nodes."""
        state_tree = state_tree if state_tree is not None else {}
        self.state_tree = state_tree
        self._flat_state = Commutator.flatten_state_tree(state_tree)

    @classmethod
    def flatten_state_tree(cls, state_tree: StateType, flat_state=None) -> StateType:
        """Recursively flatten the tree into a flat dictionary without prefixes."""
        flat_state = flat_state if flat_state is not None else {}
        for key, value in state_tree.items():
            if isinstance(value, dict):
                cls.flatten_state_tree(value, flat_state)
            else:
                flat_state[key] = value
        return flat_state

    def _update_state_tree(
        self, state_tree: StateType, key: str, value: StateType
    ) -> bool:
        """Recursively update the nested state tree."""
        for k, v in state_tree.items():
            if isinstance(v, dict):
                if key in v:
                    v[key] = value
                    return True
                if self._update_state_tree(v, key, value):
                    return True
            elif k == key:
                state_tree[k] = value
                return True
        return False

    def __setitem__(self, node_name: str, state: StateType) -> None:
        """Allow bracket notation for setting node state."""
        self.flat_state[node_name] = state
        self._update_state_tree(self.state_tree, node_name, state)

    @property
    def flat_state(self) -> StateType:
        return self._flat_state

    def get(self, node_name) -> Optional[StateType]:
        """Retrieve node state by name from the flat dictionary."""
        return self.flat_state.get(node_name)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self[key] = value

    def to_dict(self) -> StateType:
        """Convert the Commutator to a flattened dictionary."""
        return self.flat_state

    def __getitem__(self, node_name: str) -> StateType:
        """Allow bracket notation for getting node state."""
        return self.flat_state.get(node_name)

    def __contains__(self, node_name: str) -> bool:
        """Allow `in` keyword to check if node is in Commutator."""
        return node_name in self.flat_state

    def items(self):
        return self.flat_state.items()


"""commutator = Commutator(
    {
        "node_1": "state1",
        "node_2": "state2",
        "composed_1": {"node_3": "state3", "node_4": "state4"},
    }
)

print(commutator.get("node_3"))  # Output: state3
print(commutator.get("node_5"))  # Output: None
print(commutator["node_1"])  # Output: state1

# Using `in` keyword
print("node_3" in commutator)  # Output: True
print("node_5" in commutator)  # Output: False
"""
