from rcognita.systems import System, ThreeWheeledRobot
from rcognita.__utilities import rc


class ThreeWheeledRobotNI(System):
    """
    System class: 3-wheel robot with static actuators (the NI - non-holonomic integrator).
    """

    _name = "three-wheeled-robot-ni"
    _system_type = "diff_eqn"
    _dim_state = 3
    _dim_inputs = 2
    _dim_observation = 3

    def compute_state_dynamics(self, time, state, inputs):
        Dstate = rc.zeros(self.dim_state, prototype=(state, inputs))

        Dstate[0] = inputs[0] * rc.cos(state[2])
        Dstate[1] = inputs[0] * rc.sin(state[2])
        Dstate[2] = inputs[1]

        return Dstate


class Integrator(System):
    _name = "integral-parts"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _parameters = {"m": 10, "I": 1}

    def compute_state_dynamics(self, time, state, inputs):
        Dstate = rc.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, I = self.parameters["m"], self.parameters["I"]

        Dstate[0] = 1 / m * inputs[0]
        Dstate[1] = 1 / I * inputs[1]

        return Dstate


system = (
    Integrator()
    .compose(ThreeWheeledRobotNI(), output_mode="state")
    .permute_state([3, 4, 0, 1, 2])
)

print(system.name)
