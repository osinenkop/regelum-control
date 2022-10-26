"""
This module contains high-level structures of controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from .utilities import rc
import numpy as np

import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from abc import ABC, abstractmethod


class OptimalController(ABC):
    """
    A blueprint of optimal controllers.
    """

    def __init__(
        self,
        time_start=0,
        sampling_time=0.1,
        observation_target=[],
        is_fixed_critic_weights=False,
    ):

        self.controller_clock = time_start
        self.sampling_time = sampling_time

        self.observation_target = observation_target
        self.is_fixed_critic_weights = is_fixed_critic_weights
        self.new_cycle_eps_tollerance = 1e-6

    def estimate_model(self, observation, time):
        if self.is_est_model or self.mode in ["RQL", "SQL"]:
            self.estimator.estimate_model(observation, time)

    def compute_action_sampled(self, time, observation, constraints=()):

        time_in_sample = time - self.controller_clock
        timeInCriticPeriod = time - self.critic_clock
        is_critic_update = (
            timeInCriticPeriod >= self.critic_period - self.new_cycle_eps_tollerance
        ) and not self.is_fixed_critic_weights

        if is_critic_update:
            self.critic_clock = time

        if (
            time_in_sample >= self.sampling_time - self.new_cycle_eps_tollerance
        ):  # New sample
            # Update controller's internal clock
            self.controller_clock = time

            action = self.compute_action(
                time, observation, is_critic_update=is_critic_update
            )

            return action

        else:
            return self.actor.action_old

    @abstractmethod
    def compute_action(self):
        pass


class RLController(OptimalController):
    """
    Reinforcement learning controller class.
    Takes instances of `actor` and `critic` to operate.
    Action computation is sampled, i.e., actions are computed at discrete, equi-distant moments in time.
    `critic` in turn is updated every `critic_period` units of time.
    """

    def __init__(
        self, *args, critic_period=0.1, actor=[], critic=[], time_start=0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.actor = actor
        self.critic = critic

        self.dim_input = self.actor.dim_input
        self.dim_output = self.actor.dim_output

        self.critic_clock = time_start
        self.critic_period = critic_period

    def reset(self, time_start):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.

        """
        self.controller_clock = time_start
        self.critic_clock = time_start
        self.actor.action_old = self.actor.action_init

    def compute_action(
        self, time, observation, is_critic_update=False,
    ):
        # Critic

        # Update data buffers
        self.critic.update_buffers(observation, self.actor.action_old)

        if is_critic_update:
            # Update critic's internal clock
            self.critic_clock = time
            self.critic.update(time=time)

        self.actor.update(observation)
        action = self.actor.action

        return action


class NominalController3WRobot:
    """
    This is a class of nominal controllers for 3-wheel robots used for benchmarking of other controllers.

    The controller is sampled.

    For a 3-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here
    a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_]).

    Attributes
    ----------
    m, I : : numbers
        Mass and moment of inertia around vertical axis of the robot.
    controller_gain : : number
        Controller gain.
    time_start : : number
        Initial value of the controller's internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(
        self,
        m,
        I,
        controller_gain=10,
        action_bounds=[],
        time_start=0,
        sampling_time=0.1,
    ):

        self.m = m
        self.I = I
        self.controller_gain = controller_gain
        self.action_bounds = action_bounds
        self.controller_clock = time_start
        self.sampling_time = sampling_time

        self.action_old = rc.zeros(2)

    def reset(self, time_start):

        """
        Resets controller for use in multi-episode simulation.

        """
        self.controller_clock = time_start
        self.action_old = rc.zeros(2)

    def _zeta(self, xNI, theta):

        """
        Generic, i.e., theta-dependent, supper_bound_constraintradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators).

        """

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        nablaF = rc.zeros(3)

        nablaF[0] = (
            4 * xNI[0] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.cos(theta) / sigma_tilde ** 3
        )

        nablaF[1] = (
            4 * xNI[1] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.sin(theta) / sigma_tilde ** 3
        )

        nablaF[2] = (
            (
                3 * xNI[0] * rc.cos(theta)
                + 3 * xNI[1] * rc.sin(theta)
                + 2 * rc.sqrt(rc.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * rc.sign(xNI[2])
            / sigma_tilde ** 3
        )

        return nablaF

    def _kappa(self, xNI, theta):

        """
        Stabilizing controller for NI-part.

        """
        kappa_val = rc.zeros(2)

        G = rc.zeros([3, 2])
        G[:, 0] = [1, 0, xNI[1]]
        G[:, 1] = [0, 1, -xNI[0]]

        zeta_val = self._zeta(xNI, theta)

        kappa_val[0] = -rc.abs(rc.dot(zeta_val, G[:, 0])) ** (1 / 3) * rc.sign(
            rc.dot(zeta_val, G[:, 0])
        )
        kappa_val[1] = -rc.abs(rc.dot(zeta_val, G[:, 1])) ** (1 / 3) * rc.sign(
            rc.dot(zeta_val, G[:, 1])
        )

        return kappa_val

    def _Fc(self, xNI, eta, theta):

        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation.

        """

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + rc.sqrt(rc.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma_tilde ** 2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * rc.dot(z, z)

    def _minimizer_theta(self, xNI, eta):
        thetaInit = 0

        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)

        options = {"maxiter": 50, "disp": False}

        theta_val = minimize(
            lambda theta: self._Fc(xNI, eta, theta),
            thetaInit,
            method="trust-constr",
            tol=1e-6,
            bounds=bnds,
            options=options,
        ).x

        return theta_val

    def _Cart2NH(self, coords_Cart):

        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.
        See Section VIII.A in [[1]_].

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """

        xNI = rc.zeros(3)
        eta = rc.zeros(2)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        angle = coords_Cart[2]
        v = coords_Cart[3]
        omega = coords_Cart[4]

        xNI[0] = angle
        xNI[1] = xc * rc.cos(angle) + yc * rc.sin(angle)
        xNI[2] = -2 * (yc * rc.cos(angle) - xc * rc.sin(angle)) - angle * (
            xc * rc.cos(angle) + yc * rc.sin(angle)
        )

        eta[0] = omega
        eta[1] = (yc * rc.cos(angle) - xc * rc.sin(angle)) * omega + v

        return [xNI, eta]

    def _NH2ctrl_Cart(self, xNI, eta, uNI):

        """
        Get control for Cartesian NI from NH coordinates.
        See Section VIII.A in [[1]_].

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)


        """

        uCart = rc.zeros(2)

        uCart[0] = self.m * (
            uNI[1]
            + xNI[1] * eta[0] ** 2
            + 1 / 2 * (xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2])
        )
        uCart[1] = self.I * uNI[0]

        return uCart

    def compute_action_sampled(self, time, observation):
        """
        See algorithm description in [[1]_], [[2]_].

        **This algorithm needs full-state measurement of the robot**.

        References
        ----------
        .. [1] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
               via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

        .. [2] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

        """

        time_in_sample = time - self.controller_clock

        if time_in_sample >= self.sampling_time:  # New sample
            # Update internal clock
            self.controller_clock = time

            # This controller needs full-state measurement
            action = self.compute_action(observation)

            if self.action_bounds.any():
                for k in range(2):
                    action[k] = np.clip(
                        action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
                    )

            self.action_old = action

            # DEBUG ===================================================================
            # ================================LF debugger
            # R  = '\033[31m'
            # Bl  = '\033[30m'
            # headerRow = ['L']
            # dataRow = [self.compute_LF(observation)]
            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')
            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
            # print(R+table+Bl)
            # /DEBUG ===================================================================

            return action

        else:
            return self.action_old

    def compute_action(self, observation):
        """
        Same as :func:`~NominalController3WRobot.compute_action`, but without invoking the internal clock.

        """

        xNI, eta = self._Cart2NH(observation)
        theta_star = self._minimizer_theta(xNI, eta)
        kappa_val = self._kappa(xNI, theta_star)
        z = eta - kappa_val
        uNI = -self.controller_gain * z
        action = self._NH2ctrl_Cart(xNI, eta, uNI)

        self.action_old = action

        return action

    def compute_LF(self, observation):

        xNI, eta = self._Cart2NH(observation)
        theta_star = self._minimizer_theta(xNI, eta)

        return self._Fc(xNI, eta, theta_star)


class NominalController3WRobotNI:
    """
    Nominal parking controller for NI using disassembled supper_bound_constraintradients.

    """

    def __init__(
        self, controller_gain=10, action_bounds=[], time_start=0, sampling_time=0.1
    ):

        self.controller_gain = controller_gain
        self.action_bounds = action_bounds
        self.controller_clock = time_start
        self.sampling_time = sampling_time

        self.action_old = rc.zeros(2)

    def reset(self, time_start):

        """
        Resets controller for use in multi-episode simulation.

        """
        self.controller_clock = time_start
        self.action_old = rc.zeros(2)

    def _zeta(self, xNI):

        """
        Analytic disassembled supper_bound_constraintradient, without finding minimizer theta.

        """

        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(abs(xNI[2]))

        nablaL = rc.zeros(3)

        nablaL[0] = (
            4 * xNI[0] ** 3
            + rc.abs(xNI[2]) ** 3
            / sigma ** 3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[0]
        )
        nablaL[1] = (
            4 * xNI[1] ** 3
            + rc.abs(xNI[2]) ** 3
            / sigma ** 3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[1]
        )
        nablaL[2] = 3 * rc.abs(xNI[2]) ** 2 * rc.sign(xNI[2]) + rc.abs(
            xNI[2]
        ) ** 3 / sigma ** 3 * 1 / np.sqrt(rc.abs(xNI[2])) * rc.sign(xNI[2])

        theta = 0

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        nablaF = rc.zeros(3)

        nablaF[0] = (
            4 * xNI[0] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.cos(theta) / sigma_tilde ** 3
        )
        nablaF[1] = (
            4 * xNI[1] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.sin(theta) / sigma_tilde ** 3
        )
        nablaF[2] = (
            (
                3 * xNI[0] * rc.cos(theta)
                + 3 * xNI[1] * rc.sin(theta)
                + 2 * np.sqrt(rc.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * rc.sign(xNI[2])
            / sigma_tilde ** 3
        )

        if xNI[0] == 0 and xNI[1] == 0:
            return nablaF
        else:
            return nablaL

    def _kappa(self, xNI):

        """
        Stabilizing controller for NI-part.

        """
        kappa_val = rc.zeros(2)

        G = rc.zeros([3, 2])
        G[:, 0] = rc.array([1, 0, xNI[1]], prototype=G)
        G[:, 1] = rc.array([0, 1, -xNI[0]], prototype=G)

        zeta_val = self._zeta(xNI)

        kappa_val[0] = -rc.abs(np.dot(zeta_val, G[:, 0])) ** (1 / 3) * rc.sign(
            rc.dot(zeta_val, G[:, 0])
        )
        kappa_val[1] = -rc.abs(np.dot(zeta_val, G[:, 1])) ** (1 / 3) * rc.sign(
            rc.dot(zeta_val, G[:, 1])
        )

        return kappa_val

    def _F(self, xNI, eta, theta):

        """
        Marginal function for NI.

        """

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma_tilde ** 2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * np.dot(z, z)

    def _Cart2NH(self, coords_Cart):

        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.

        """

        xNI = rc.zeros(3)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        angle = coords_Cart[2]

        xNI[0] = angle
        xNI[1] = xc * rc.cos(angle) + yc * rc.sin(angle)
        xNI[2] = -2 * (yc * rc.cos(angle) - xc * rc.sin(angle)) - angle * (
            xc * rc.cos(angle) + yc * rc.sin(angle)
        )

        return xNI

    def _NH2ctrl_Cart(self, xNI, uNI):

        """
        Get control for Cartesian NI from NH coordinates.

        """

        uCart = rc.zeros(2)

        uCart[0] = uNI[1] + 1 / 2 * uNI[0] * (xNI[2] + xNI[0] * xNI[1])
        uCart[1] = uNI[0]

        return uCart

    def compute_action_sampled(self, time, observation):
        """
        Compute sampled action.

        """

        time_in_sample = time - self.controller_clock

        if time_in_sample >= self.sampling_time:  # New sample
            # Update internal clock
            self.controller_clock = time

            action = self.compute_action(observation)

            if self.action_bounds.any():
                for k in range(2):
                    action[k] = np.clip(
                        action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
                    )

            self.action_old = action

            # DEBUG ===================================================================
            # ================================LF debugger
            # R  = '\033[31m'
            # Bl  = '\033[30m'
            # headerRow = ['L']
            # dataRow = [self.compute_LF(observation)]
            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')
            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
            # print(R+table+Bl)
            # /DEBUG ===================================================================

            return action

        else:
            return self.action_old

    def compute_action(self, observation):
        """
        Same as :func:`~NominalController3WRobotNI.compute_action`, but without invoking the internal clock.

        """

        xNI = self._Cart2NH(observation)
        kappa_val = self._kappa(xNI)
        uNI = self.controller_gain * kappa_val
        action = self._NH2ctrl_Cart(xNI, uNI)

        self.action_old = action

        return action

    def compute_LF(self, observation):

        xNI = self._Cart2NH(observation)

        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(rc.abs(xNI[2]))

        return xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma ** 2


class NominalControllerInvertedPendulum:
    def __init__(self, action_bounds, controller_gain):
        self.action_bounds = action_bounds
        self.controller_gain = controller_gain
        self.observation = np.array([np.pi, 0])

    def __call__(self, observation):
        return self.compute_action(observation)

    def compute_action(self, observation):
        self.observation = observation
        return np.array([-((observation[0]) + (observation[1])) * self.controller_gain])
        # return np.array(
        #     [
        #         np.clip(
        #             -((observation[0]) + (observation[1])) * self.controller_gain,
        #             self.action_bounds[0][0],
        #             self.action_bounds[0][1],
        #         )
        #     ]
        # )
        # return np.array(
        #     [
        #         np.clip(
        #             -np.sign(observation[0])
        #             * np.exp(
        #                 observation[0] + observation[1] * np.sign(observation[1]) ** 2
        #             )
        #             * self.controller_gain,
        #             self.action_bounds[0][0],
        #             self.action_bounds[0][1],
        #         )
        #     ]
        # )
