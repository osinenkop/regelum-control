"""Module that contains state or observation (depending on the context) predictors."""

from abc import ABC, abstractmethod

import regelum
from .utils import rg
from .system import System
from .model import ModelWeightContainer, ModelWeightContainerTorch
import torch


class Predictor(regelum.RegelumBase, ABC):
    """Blueprint of a predictor."""

    def __init__(
        self,
        system: System,
        pred_step_size: float,
    ):
        """Initialize an instance of a predictor.

        Args:
            system (System): System of which states are predicted
            pred_step_size (float): time interval between successive
                predictions
        """
        self.system = system
        self.pred_step_size = pred_step_size

    @abstractmethod
    def predict(self):
        pass

    def predict_state_sequence_from_action_sequence(
        self,
        state,
        action_sequence,
        is_predict_last: bool,
        return_predicted_states_only=False,
    ):
        """Predicts the state sequence from the given action sequence.

        Args:
            state (numpy.ndarray): The initial state.
            action_sequence (numpy.ndarray): The sequence of actions.
            is_predict_last (bool): Whether to predict the last state or
                not.
            return_predicted_states_only (bool, optional): Whether to
                return only the predicted states or both the predicted
                states and the action sequence. Defaults to False.

        Returns:
            tuple or numpy.ndarray: If `return_predicted_states_only` is
            False, returns a tuple containing the predicted state
            sequence and the action sequence. If
            `return_predicted_states_only` is True, returns only the
            predicted state sequence.
        """
        len_state_sequence = action_sequence.shape[0] - int(not is_predict_last)
        predicted_state_sequence = rg.zeros(
            [len_state_sequence, self.system.dim_state],
            prototype=action_sequence,
        )
        current_state = state
        for k in range(len_state_sequence):
            current_action = action_sequence[k, :]
            next_state = self.predict(current_state, current_action)
            predicted_state_sequence[k, :] = next_state
            current_state = next_state

        return (
            (predicted_state_sequence, action_sequence)
            if not return_predicted_states_only
            else predicted_state_sequence
        )

    def predict_state_sequence_from_model(
        self,
        state,
        prediction_horizon,
        is_predict_last: bool,
        model,
        model_weights=None,
        return_predicted_states_only=False,
    ):
        """Predicts a sequence of states from a given initial state using a model.

        Args:
            state: The initial state.
            prediction_horizon: The number of steps to predict into the
                future.
            is_predict_last: A boolean indicating whether to predict the
                last state.
            model: The model used for prediction.
            model_weights: The weights of the model (optional).
            return_predicted_states_only: A boolean indicating whether
                to return only the predicted states.

        Returns:
            If `return_predicted_states_only` is `True`, returns the
            predicted state sequence. Otherwise, returns a tuple
            containing the predicted state sequence and the action
            sequence.
        """
        if isinstance(model, ModelWeightContainer):
            if model_weights is not None:
                assert model_weights.shape[0] == prediction_horizon + 1 - int(
                    is_predict_last
                ), f"model_weights.shape[0] = {model_weights.shape[0]} should have length prediction_horizon + 1 - int(is_predict_last) = {prediction_horizon + 1 - int(is_predict_last)}"
                return self.predict_state_sequence_from_action_sequence(
                    state,
                    action_sequence=model_weights,
                    is_predict_last=is_predict_last,
                    return_predicted_states_only=return_predicted_states_only,
                )
            else:
                assert model._weights.shape[0] == prediction_horizon + 1 - int(
                    is_predict_last
                ), "model._weights.shape[0] should have length prediction_horizon + 1 - int(is_predict_last)"
                return self.predict_state_sequence_from_action_sequence(
                    state,
                    action_sequence=model._weights,
                    is_predict_last=is_predict_last,
                    return_predicted_states_only=return_predicted_states_only,
                )
        elif isinstance(model, ModelWeightContainerTorch):
            assert model._weights.shape[0] == prediction_horizon + 1 - int(
                is_predict_last
            ), "model._weights.shape[0] should have length prediction_horizon + 1 - int(is_predict_last)"
            dummy_input = torch.zeros(
                [
                    prediction_horizon + 1 - int(is_predict_last),
                    self.system.dim_observation,
                ],
            )
            return self.predict_state_sequence_from_action_sequence(
                state,
                action_sequence=model(dummy_input),
                is_predict_last=is_predict_last,
                return_predicted_states_only=return_predicted_states_only,
            )

        predicted_state_sequence = rg.zeros(
            [prediction_horizon + 1 - int(not is_predict_last), self.system.dim_state],
            prototype=state,
        )
        action_sequence = rg.zeros(
            [prediction_horizon + 1, self.system.dim_state],
            prototype=state,
        )
        current_state = state
        for k in range(prediction_horizon + 1):
            action_sequence[k, :] = model(current_state, model_weights)
            next_state = self.predict(current_state, action_sequence[k, :])
            if k < predicted_state_sequence.shape[0]:
                predicted_state_sequence[k, :] = next_state

            current_state = next_state
        return (
            (predicted_state_sequence, action_sequence)
            if not return_predicted_states_only
            else predicted_state_sequence
        )


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


class EulerPredictorMultistep(EulerPredictor):
    """Applies several iterations of Euler estimation to predict a single step."""

    def __init__(self, *args, n_steps=5, **kwargs):
        """Initialize an instance of EulerPredictorMultistep.

        Args:
            *args: positional arguments for EulerPredictor
            n_steps: number of estimations to predict a single step
            **kwargs: keyword arguments for EulerPredictor
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

        Args:
            system: an instance of a discrete system
        """
        self.system = system

    def predict(self, time, state, action):
        return self.system.compute_state_dynamics(time, state, action)
