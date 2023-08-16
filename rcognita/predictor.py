"""Module that contains state or observation (depending on the context) predictors."""

from abc import ABC, abstractmethod

import rcognita
from .__utilities import rc
from .system import System


class Predictor(rcognita.RcognitaBase, ABC):
    """Blueprint of a predictor."""

    def __init__(
        self,
        system: System,
        pred_step_size: float,
        prediction_horizon: int,
    ):
        """Initialize an instance of a predictor.

        :param system: System of which states are predicted
        :type system: System
        :param pred_step_size: time interval between successive predictions
        :type pred_step_size: float
        :param prediction_horizon: number of predictions
        :type prediction_horizon: int
        """
        self.system = system
        self.pred_step_size = pred_step_size
        self.prediction_horizon = prediction_horizon

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_sequence(self):
        pass


class EulerPredictor(Predictor):
    """Euler predictor uses a simple Euler discretization scheme.

    It does predictions by increments scaled by a sampling time times the velocity evaluated at each successive node.

    """

    def predict(self, current_state, action):
        next_state = (
            current_state
            + self.pred_step_size
            * self.system.compute_state_dynamics(
                time=None, state=current_state, inputs=action
            )
        )

        return next_state

    def predict_sequence(self, state, action_sequence):
        state_sequence = rc.zeros(
            [self.prediction_horizon, self.system.dim_state], prototype=action_sequence
        )
        current_state = state

        for k in range(self.prediction_horizon):
            current_action = action_sequence[k, :]
            next_state = self.predict(current_state, current_action)
            state_sequence[k, :] = self.system.get_observation(
                time=None, state=next_state, inputs=current_action
            )
            current_state = next_state
        return state_sequence


class EulerPredictorMultistep(EulerPredictor):
    """Applies several iterations of Euler estimation to predict a single step."""

    def __init__(self, *args, n_steps=5, **kwargs):
        """Initialize an instance of EulerPredictorMultistep.

        :param args: positional arguments for EulerPredictor
        :param n_steps: number of estimations to predict a single step
        :param kwargs: keyword arguments for EulerPredictor
        """
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.pred_step_size /= self.n_steps

    def predict(self, current_state_or_observation, action):
        next_state_or_observation = current_state_or_observation
        for _ in range(self.n_steps):
            next_state_or_observation = super().predict(
                next_state_or_observation, action
            )
        return next_state_or_observation


class TrivialPredictor(Predictor):
    """A predictor that propagates the observation or state directly through the system dynamics law."""

    def __init__(self, system):
        """Initialize an instance of TrivialPredictor.

        :param system: an instance of a discrete system
        """
        self.system = system

    def predict(self, time, state, action):
        return self.system.compute_state_dynamics(time, state, action)

    def predict_sequence(self, time, state, action):
        return self.predict(time, state, action)
