from .systems import System
from .__utilities import rc
from abc import ABC, abstractmethod


class TwoTankPlanner(System):
    _name = "two-tank"
    _system_type = "diff_eqn"
    _dim_state = 0
    _dim_inputs = 2
    _dim_observation = 2
    observation_target = [0.4, 0.4]

    def compute_state_dynamics(self, time, state, inputs):
        return rc.zeros(self.dim_state, prototype=(state, inputs))

    def get_observation(self, time, state, inputs):
        return inputs - rc.array(self.observation_target, prototype=state)
