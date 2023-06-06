"""
Module that contains state or observation (depending on the context) predictors.

"""

from abc import ABC, abstractmethod

import rcognita.base
from .__utilities import rc
from .systems import System


class Predictor(rcognita.base.RcognitaBase, ABC):
    """
    Blueprint of a predictor.

    """

    def __init__(
        self,
        system: System,
        pred_step_size: float,
        prediction_horizon: int,
    ):
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
    """
    Euler predictor uses a simple Euler discretization scheme.
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
            [self.system.dim_state, self.prediction_horizon], prototype=action_sequence
        )
        current_state = state

        for k in range(self.prediction_horizon):
            current_action = action_sequence[:, k]
            next_state = self.predict(current_state, current_action)
            state_sequence[:, k] = rc.transpose(
                self.system.get_observation(
                    time=None, state=next_state, inputs=current_action
                )
            )
            current_state = next_state
        return state_sequence


class EulerPredictorMultistep(EulerPredictor):
    def __init__(self, *args, n_steps=5, **kwargs):
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


class RKPredictor(EulerPredictor):
    """
    Predictor that makes use o Runge-Kutta finite difference methods.
    """

    def __init__(
        self,
        state_or_observation_init,
        action_init,
        *args,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.atol = atol
        self.rtol = rtol
        self.integrator = self.create_CasADi_integrator(
            self.system,
            self.pred_step_size,
        )

    def create_CasADi_integrator(self, system, max_step):
        try:
            import casadi
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Cannot use RKPredictor without casadi being installed"
            )

        state_symbolic = rc.array_symb(self.system.dim_state, literal="x")
        action_symbolic = rc.array_symb(self.system.dim_inputs, literal="u")
        time = rc.array_symb((1, 1), literal="t")

        ODE = system.compute_state_dynamics(time, state_symbolic, action_symbolic)
        DAE = {"x": state_symbolic, "p": action_symbolic, "ode": ODE}
        options = {"tf": max_step, "atol": self.atol, "rtol": self.rtol}

        integrator = casadi.integrator("intg", "rk", DAE, options)

        return integrator

    def predict(self, current_state_or_observation, action):
        state_new = self.integrator(x0=current_state_or_observation, p=action)["xf"]
        try:
            state_new = rc.squeeze(state_new.full().T)
        except:
            pass

        return state_new


class TrivialPredictor(Predictor):
    """
    This predictor propagates the observation or state directly through the system dynamics law.

    """

    def __init__(self, system):
        self.system = system

    def predict(self, time, state, action):
        return self.system.compute_state_dynamics(time, state, action)

    def predict_sequence(self, time, state, action):
        return self.predict(time, state, action)
