from regelum import rg
from regelum.environment.commutator import Commutator
from regelum.environment.transistor import CasADi
from regelum.environment.node import Node, Terminate, Clock, ComposedNode, State, Inputs
from regelum.environment.graph_builder import GraphBuilder


class InvPendulum(Node):
    state = State("plant", (2, 1))
    inputs = Inputs({"action": (1, 1)})

    def compute_state_dynamics(self, state, inputs):

        Dstate = rg.zeros(
            self.state.dims,
            prototype=(state),
        )
        Dstate[0] = state[1]
        Dstate[1] = rg.sin(state[0]) + inputs["action"][0]
        return Dstate


inv_pendulum_node = InvPendulum(
    state_init=rg.DM([3.14, 0.0]),
    transistor=CasADi(10.0, 0.01, time_start=0.0),
    is_root=True,
)

clock = Clock([inv_pendulum_node])
terminate_inv_pendulum_node = Terminate(inv_pendulum_node)

# whole_node = ComposedNode([inv_pendulum_node, clock, terminate_inv_pendulum_node])

# commutator = dict()  # {"action": rg.array([1.0])}
# print(commutator)
# new_state = whole_node.transistor.step(commutator)
# print(new_state)

part_node = ComposedNode([inv_pendulum_node, clock], is_root=True)
whole_node = ComposedNode([part_node, terminate_inv_pendulum_node])

commutator = Commutator({})
print(commutator.flat_state)
whole_node.transistor.step(commutator)
commutator["action"] = rg.array([1.0])
print(commutator.flat_state)
whole_node.transistor.step(commutator)
print(commutator.flat_state)
