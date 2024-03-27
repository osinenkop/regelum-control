"""Tools for systems' state estimation."""

import numpy as np
from abc import ABC, abstractmethod

import regelum
from typing import Union, List


class Observer(regelum.RegelumBase, ABC):
    """A class implementing observer."""

    @abstractmethod
    def get_state_estimation(self): ...


class ObserverTrivial(Observer):
    """A class implementing a trivial observer, which simply returns the current state."""

    def get_state_estimation(self, t, observation, action):
        return observation


class ObserverReference(Observer):
    """Estimates state via adding reference to observation."""

    def __init__(self, reference: Union[np.ndarray, List[float]]):
        """Instatiate ObserverReference.

        Args:
            reference (Union[np.ndarray, List[float]]): array for
                reference
        """
        self.reference = np.array(reference).reshape(1, -1)

    def get_state_estimation(self, t, observation, action):
        return observation + self.reference


class CartPoleObserverPG(Observer):
    def __init__(self):
        pass

    def get_state_estimation(self, t, observation: np.ndarray, action):
        # sin_theta, one_minus_cos_theta, x, theta_dot, x_dot = observation.reshape(-1)
        sin_theta, one_minus_cos_theta, theta_dot, x_dot = observation.reshape(-1)

        return np.array(
            [[np.arctan2(sin_theta, 1 - one_minus_cos_theta), 0, theta_dot, x_dot]]
        )


class CartPoleObserver(Observer):
    def __init__(self):
        pass

    def get_state_estimation(self, t, observation: np.ndarray, action):
        sin_theta, one_minus_cos_theta, x, theta_dot, x_dot = observation.reshape(-1)
        # sin_theta, one_minus_cos_theta, theta_dot, x_dot = observation.reshape(-1)

        return np.array(
            [[np.arctan2(sin_theta, 1 - one_minus_cos_theta), x, theta_dot, x_dot]]
        )


class KalmanFilter(Observer):
    """A class implementing Kalman filter."""

    def __init__(
        self,
        t0,
        my_sys,
        sys_noise_cov,
        observ_noise_cov,
        prior_est_cov,
        state_init,
    ):
        """Initialize an instance of KalmanFilter.

        Args:
            t0: time at which simulation starts
            my_sys: an instance of a system of which state is to be
                observed
            sys_noise_cov: system noise covariance matrix
            observ_noise_cov: observation noise covariance matrix
            prior_est_cov: prior esimation covariance
            state_init: initial state
        """
        self.my_sys = my_sys

        self.posterior_state_est = state_init
        self.prior_state_est = None
        self.dim_state = self.posterior_state_est.shape[0]
        self.sys_noise_cov = sys_noise_cov
        self.observ_noise_cov = observ_noise_cov

        self.posterior_est_cov = np.eye(self.dim_state)
        self.prior_est_cov = prior_est_cov

        self.est_clock = t0

    def predict_state(self, action, dt):
        Q = self.sys_noise_cov
        J = self.my_sys.Jacobi_system_matrix(self.posterior_state_est)
        P_posterior_prev = self.posterior_est_cov

        self.prior_state_est = self.posterior_state_est + dt * self.my_sys._state_dyn(
            [], self.posterior_state_est, action
        )

        self.prior_est_cov = J @ P_posterior_prev @ J.T + Q

    def correct_state(self, observation):
        z = np.array(observation)
        J_h = self.my_sys.Jacobi_observation_matrix()
        P_pred = self.prior_est_cov
        R = self.observ_noise_cov

        K = np.array(P_pred @ J_h.T @ np.linalg.inv(J_h @ P_pred @ J_h.T + R))
        self.posterior_state_est = self.prior_state_est + K @ (
            z
            - self.my_sys.get_observation(
                time=None, state=self.prior_state_est, inputs=None
            )
        )
        self.posterior_est_cov = (np.eye(self.dim_state) - K @ J_h) @ P_pred
        print(f"Kalman gain:\n{K}, \n Post est cov: \n{self.posterior_est_cov}")

    def get_state_estimation(self, t, observation, action):
        dt = t - self.est_clock
        self.est_clock = t
        print(f"TRACE OF COV EST MATRIX:{self.prior_est_cov.trace()}\n")
        self.predict_state(action, dt)
        self.correct_state(observation)
        return self.posterior_state_est
