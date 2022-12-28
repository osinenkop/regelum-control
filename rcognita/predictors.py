"""
Module that contains state or observation (depending on the context) predictors.

"""

import numpy as np
from abc import ABC, abstractmethod

from .__utilities import rc
from .systems import System
from .solvers import create_CasADi_integrator


class Predictor(ABC):
    """
    Blueprint of a predictor.

    """

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_sequence(self):
        pass


class EulerPredictor(Predictor):
    """
    Euler predictor uses a simple Euler discretization scheme.
    It does predictions by increments scaled by a sampling time times the velocity evaluated at each successive node.

    """

    def __init__(
        self,
        pred_step_size: float,
        system: System,
        dim_input: int,
        prediction_horizon: int,
    ):
        self.system = system
        self.pred_step_size = pred_step_size
        self.compute_state_dynamics = system.compute_dynamics
        self.sys_out = system.out
        self.dim_input = dim_input
        self.prediction_horizon = prediction_horizon

    def predict(self, current_state_or_observation, action):
        next_state_or_observation = (
            current_state_or_observation
            + self.pred_step_size
            * self.compute_state_dynamics([], current_state_or_observation, action)
        )
        return next_state_or_observation

    def predict_sequence(self, observation, action_sequence):

        observation_sequence = rc.zeros(
            [self.dim_input, self.prediction_horizon], prototype=action_sequence
        )
        current_observation = observation

        for k in range(self.prediction_horizon):
            current_action = action_sequence[:, k]
            next_observation = self.predict(current_observation, current_action)
            observation_sequence[:, k] = self.sys_out(next_observation).T
            current_observation = next_observation
        return observation_sequence


class EulerPredictorPendulum(EulerPredictor):
    def predict(self, current_state_or_observation, action):
        rhs = self.compute_state_dynamics([], current_state_or_observation, action)
        next_state_or_observation = (
            current_state_or_observation
            + self.pred_step_size * rc.array([rhs[0], 0, rhs[1]])
        )
        return next_state_or_observation

    def predict_sequence(self, observation, action_sequence):

        observation_sequence = rc.zeros(
            [self.prediction_horizon, self.dim_output], prototype=action_sequence
        )
        current_observation = observation

        for k in range(self.prediction_horizon):
            current_action = action_sequence[k, :]
            next_observation = self.predict(current_observation, current_action)
            observation_sequence[k, :] = self.sys_out(next_observation)
            current_observation = next_observation
        return observation_sequence


class RKPredictor(EulerPredictor):
    """
    Predictor that makes use o Runge-Kutta finite difference methods.
    """

    def __init__(self, state_or_observation_init, action_init, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.integrator = create_CasADi_integrator(
            self.system, state_or_observation_init, action_init, self.pred_step_size,
        )

    def predict(self, current_state_or_observation, action):
        state_new = self.integrator(x0=current_state_or_observation, p=action)["xf"]
        return state_new


class TrivialPredictor(Predictor):
    """
    This predictor propagates the observation or state directly through the system dynamics law.

    """

    def __init__(self, system):
        self.compute_dynamics = system.compute_dynamics

    def predict(self, current_state_or_observation, action):
        return self.compute_dynamics(current_state_or_observation, action)

    def predict_sequence(self, current_state_or_observation, action):
        return self.predict(current_state_or_observation, action)
