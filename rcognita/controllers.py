"""
This module contains high-level structures of controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize

from .__utilities import rc, Clock
from .optimizers import CasADiOptimizer, SciPyOptimizer
from .__w_plotting import plot_optimization_results


class Controller(ABC):
    """
    A blueprint of optimal controllers.
    """

    def __init__(
        self,
        time_start: float = 0,
        sampling_time: float = 0.1,
        is_fixed_critic_weights: bool = False,
    ):

        self.controller_clock = time_start
        self.sampling_time = sampling_time

        self.observation_target = []
        self.is_fixed_critic_weights = is_fixed_critic_weights
        self.clock = Clock(period=sampling_time, time_start=time_start)

    def compute_action_sampled(self, time, observation, constraints=()):

        is_time_for_new_sample = self.clock.check_time(time)
        is_time_for_critic_update = self.critic.clock.check_time(time)

        is_critic_update = (
            is_time_for_critic_update and not self.is_fixed_critic_weights
        )

        if is_time_for_new_sample:  # New sample
            # Update controller's internal clock

            self.compute_action(time, observation, is_critic_update=is_critic_update)

        return self.actor.action

    @abstractmethod
    def compute_action(self):
        pass


class ControllerStabilizing(ABC):
    """
    A blueprint of optimal controllers.
    """

    def __init__(self, *args, observation_target: list = None, **kwargs):

        super().__init__(*args, **kwargs)
        if observation_target is None:
            observation_target = []

        self.observation_target = observation_target


class RLController(Controller):
    """
    Reinforcement learning controller class.
    Takes instances of `actor` and `critic` to operate.
    Action computation is sampled, i.e., actions are computed at discrete, equi-distant moments in time.
    `critic` in turn is updated every `critic_period` units of time.
    """

    def __init__(
        self,
        *args,
        critic_period=0.1,
        actor=None,
        critic=None,
        time_start=0,
        action_bounds=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.actor = actor
        self.critic = critic
        self.action_bounds = action_bounds

        self.critic_clock = time_start
        self.critic_period = critic_period
        self.weights_difference_norms = []

    def reset(self, time_start):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.

        """
        self.clock.reset()
        self.critic.clock.reset()
        self.actor.action_old = self.actor.action_init

    def compute_action(
        self, time, observation, is_critic_update=True,
    ):
        ### store current action and observation in critic's data buffer
        self.critic.update_buffers(observation, self.actor.action)

        ### store current observation in actor
        self.actor.receive_observation(observation)

        if is_critic_update:
            ### optimize critic's model weights
            self.critic.optimize_weights(time=time)
            ### substitute and cache critic's model optimized weights
            self.critic.update_and_cache_weights()

        ### optimize actor's model weights based on current observation
        self.actor.optimize_weights()

        ### substitute and cache weights in the actor's model
        self.actor.update_and_cache_weights()
        self.actor.update_action(observation)

        return self.actor.action


class CALFControllerExPost(RLController):
    def compute_weights_displacement(self, agent):
        self.weights_difference_norm = rc.norm_2(
            self.critic.model.cache.weights - self.critic.optimized_weights
        )
        self.weights_difference_norms.append(self.weights_difference_norm)

    def invoke_safe_action(self, observation):
        # self.actor.restore_weights()
        # self.critic.restore_weights()
        action = self.actor.safe_controller.compute_action(observation)

        self.actor.set_action(action)
        self.actor.model.update_and_cache_weights(action)

    def compute_action(
        self, time, observation, is_critic_update=False,
    ):
        # Update data buffers
        self.critic.update_buffers(
            observation, self.actor.action
        )  ### store current action and observation in critic's data buffer

        # self.critic.safe_decay_rate = 1e-1 * rc.norm_2(observation)
        self.actor.receive_observation(
            observation
        )  ### store current observation in actor

        self.critic.optimize_weights(time=time)

        critic_weights_accepted = self.critic.weights_acceptance_status == "accepted"

        if critic_weights_accepted:
            self.critic.update_weights()

            self.invoke_safe_action(observation)

            self.actor.optimize_weights(time=time)
            actor_weights_accepted = self.actor.weights_acceptance_status == "accepted"

            if actor_weights_accepted:
                self.actor.update_and_cache_weights()
                self.actor.update_action()

                self.critic.observation_last_good = observation
                self.critic.cache_weights()
            else:
                self.invoke_safe_action(observation)
        else:
            self.invoke_safe_action(observation)

        self.collect_critic_stats(time)

        return self.actor.action

    def collect_critic_stats(self, time):
        self.critic.stabilizing_constraint_violations.append(
            np.squeeze(self.critic.stabilizing_constraint_violation)
        )
        self.critic.lb_constraint_violations.append(
            0
        )  # (self.critic.lb_constraint_violation)
        self.critic.ub_constraint_violations.append(
            0
        )  # (self.critic.ub_constraint_violation)
        self.critic.Ls.append(
            np.squeeze(
                self.critic.safe_controller.compute_LF(self.critic.current_observation)
            )
        )
        self.critic.times.append(time)
        current_CALF = self.critic(
            self.critic.observation_last_good, use_stored_weights=True
        )
        self.critic.values.append(
            np.squeeze(self.critic.model(self.critic.current_observation))
        )
        if self.critic.CALFs != []:
            CALF_increased = current_CALF > self.critic.CALFs[-1]
            if CALF_increased:
                print("CALF increased!!")

        print(self.critic.model.weights, time)

        self.critic.CALFs.append(current_CALF)


class CALFControllerPredictive(CALFControllerExPost):
    def compute_action(
        self, time, observation, is_critic_update=False,
    ):

        # Update data buffers
        self.critic.update_buffers(
            observation, self.actor.action
        )  ### store current action and observation in critic's data buffer
        # if on prev step weifhtts were acccepted, then upd last good
        if self.actor.weights_acceptance_status == "accepted":
            self.critic.observation_last_good = observation
            self.critic.weights_acceptance_status = "rejected"
            self.actor.weights_acceptance_status = "rejected"
            if self.critic.CALFs != []:
                self.critic.CALFs[-1] = self.critic(
                    self.critic.observation_last_good, use_stored_weights=True
                )

        # Store current observation in actor
        self.actor.receive_observation(observation)

        self.critic.optimize_weights(time=time)

        if self.critic.weights_acceptance_status == "accepted":
            self.critic.update_weights()

            self.invoke_safe_action(observation)

            self.actor.optimize_weights(time=time)

            if self.actor.weights_acceptance_status == "accepted":
                self.actor.update_and_cache_weights()
                self.actor.update_action()

                self.critic.cache_weights()
            else:
                self.invoke_safe_action(observation)
        else:
            self.invoke_safe_action(observation)

        self.collect_critic_stats(time)

        # plot_optimization_results(
        #     self.critic.cost_function,
        #     self.critic.constraint,
        #     self.actor.cost_function,
        #     self.actor.constraint,
        #     self.critic.symbolic_var,
        #     self.actor.symbolic_var,
        #     self.critic.weights_init,
        #     self.critic.optimized_weights,
        #     self.actor.weights_init,
        #     self.actor.optimized_weights,
        # )

        return self.actor.action


class Controller3WRobotDisassembledCLF:
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
        action_bounds=None,
        time_start=0,
        sampling_time=0.01,
        max_iters=200,
        optimizer_engine="SciPy",
    ):

        self.m = m
        self.I = I
        self.controller_gain = controller_gain
        self.action_bounds = action_bounds
        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.clock = Clock(period=sampling_time, time_start=time_start)

        self.action_old = rc.zeros(2)

        self.optimizer_engine = optimizer_engine

        if optimizer_engine == "CasADi":
            casadi_opt_options = {
                "print_time": 0,
                "ipopt.max_iter": max_iters,
                "ipopt.print_level": 0,
                "ipopt.acceptable_tol": 1e-7,
                "ipopt.acceptable_obj_change_tol": 1e-2,
            }
            self.casadi_optimizer = CasADiOptimizer(
                opt_method="ipopt", opt_options=casadi_opt_options
            )

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

        nablaF = rc.zeros(3, prototype=theta)

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

        G = rc.zeros([3, 2])
        G[:, 0] = [1, 0, xNI[1]]
        G[:, 1] = [0, 1, -xNI[0]]

        kappa_val = rc.zeros(2, prototype=theta)

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

        objective_lambda = lambda theta: self._Fc(xNI, eta, theta)
        if self.optimizer_engine == "SciPy":
            bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)
            options = {"maxiter": 50, "disp": False}
            theta_val = minimize(
                objective_lambda,
                thetaInit,
                method="trust-constr",
                tol=1e-4,
                bounds=bnds,
                options=options,
            ).x

        elif self.optimizer_engine == "CasADi":
            symbolic_var = rc.array_symb((1, 1), literal="x")
            objective_symbolic = rc.lambda2symb(objective_lambda, symbolic_var)

            theta_val = self.casadi_optimizer.optimize(
                objective=objective_symbolic,
                initial_guess=rc.array([thetaInit], rc_type=rc.CASADI),
                bounds=[-np.pi, np.pi],
                decision_variable_symbolic=symbolic_var,
            )

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
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample

            # This controller needs full-state measurement
            action = self.compute_action(observation)

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
            # if self.action_bounds.any():
            #     for k in range(2):
            #         action[k] = np.clip(
            #             action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
            #         )

            return action

        else:
            return self.action_old

    def compute_action(self, observation):
        """
        Same as :func:`~Controller3WRobotDisassembledCLF.compute_action`, but without invoking the internal clock.

        """

        xNI, eta = self._Cart2NH(observation)
        theta_star = self._minimizer_theta(xNI, eta)
        kappa_val = self._kappa(xNI, theta_star)
        z = eta - kappa_val
        uNI = -self.controller_gain * z
        action = self._NH2ctrl_Cart(xNI, eta, uNI)

        # if self.action_bounds.any():
        #     for k in range(2):
        #         action[k] = np.clip(
        #             action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
        #         )

        self.action_old = action

        return action

    def compute_LF(self, observation):

        xNI, eta = self._Cart2NH(observation)
        theta_star = self._minimizer_theta(xNI, eta)

        return self._Fc(xNI, eta, theta_star)


class ControllerPID:
    def __init__(
        self,
        P,
        I,
        D,
        SP=None,
        sampling_time=0.01,
        initial_point=(-5, -5),
        buffer_length=30,
    ):
        self.P = P
        self.I = I
        self.D = D

        self.SP = SP
        self.integral = 0.0
        self.error_old = 0.0
        self.sampling_time = sampling_time
        self.clock = Clock(period=sampling_time, time_start=0)
        self.initial_point = initial_point
        if isinstance(initial_point, (float, int)):
            self.state_size = 1
        else:
            self.state_size = len(initial_point)

        self.buffer_length = buffer_length
        self.observation_buffer = rc.ones((self.state_size, buffer_length)) * 1e3

    def compute_error(self, PV):
        if isinstance(PV, (float, int)):
            error = PV - self.SP
        else:
            if len(PV) == 1:
                error = PV - self.SP
            else:
                norm = rc.norm_2(self.SP - PV)
                error = norm * rc.sign(rc.dot(self.initial_point, PV))
        return error

    def compute_integral(self, error):
        self.integral += error * self.sampling_time
        return self.integral

    def compute_error_derivative(self, error):
        error_derivative = (error - self.error_old) / self.sampling_time / 10.0
        self.error_old = error
        return error_derivative

    def compute_action(self, PV, error_derivative=None):

        error = self.compute_error(PV)
        integral = self.compute_integral(error)

        if error_derivative is None:
            error_derivative = self.compute_error_derivative(error)

        PID_signal = self.P * error + self.I * integral + self.D * error_derivative

        ### DEBUG ==============================
        # print(error, integral, error_derivative)
        ### /DEBUG =============================

        return PID_signal

    def set_SP(self, SP):
        self.SP = SP

    def update_observation_buffer(self, observation):
        self.observation_buffer = rc.push_vec(self.observation_buffer, observation)

    def set_initial_point(self, point):
        self.initial_point = point

    def reset(self):
        self.integral = 0.0
        self.error_old = 0.0

    def reset_buffer(self):
        self.observation_buffer = rc.ones((self.state_size, self.buffer_length)) * 1e3

    def is_stabilized(self, stabilization_tollerance=1e-3):
        is_stabilized = np.allclose(
            self.observation_buffer,
            rc.rep_mat(rc.reshape(self.SP, (-1, 1)), 1, self.buffer_length),
            atol=stabilization_tollerance,
        )
        return is_stabilized


class Controller3WRobotPID:
    def __init__(
        self,
        state_init,
        params=None,
        time_start=0,
        sampling_time=0.01,
        action_bounds=None,
    ):
        if params is None:
            params = [10, 1]

        self.m, self.I = params
        if action_bounds is None:
            action_bounds = []

        self.action_bounds = action_bounds
        self.state_init = state_init

        self.controller_clock = time_start
        self.sampling_time = sampling_time

        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.Ls = []
        self.times = []
        self.action_old = rc.zeros(2)
        self.PID_angle_arctan = ControllerPID(
            -35, 0.0, -10, initial_point=self.state_init[2]
        )
        self.PID_v_zero = ControllerPID(
            -35, 0.0, 1.2, initial_point=self.state_init[3], SP=0.0
        )
        self.PID_x_y_origin = ControllerPID(
            -35,
            0.0,
            -35,
            SP=rc.array([0.0, 0.0]),
            initial_point=self.state_init[:2],
            # buffer_length=100,
        )
        self.PID_angle_origin = ControllerPID(
            -30, 0.0, -10, SP=0.0, initial_point=self.state_init[2]
        )
        self.stabilization_tollerance = 1e-3
        self.current_F = 0
        self.current_M = 0

    def get_SP_for_PID_angle_arctan(self, x, y):
        return np.arctan2(y, x)

    def compute_square_of_norm(self, x, y):
        return rc.sqrt(rc.norm_2(rc.array([x, y])))

    def compute_action(self, observation):
        x = observation[0]
        y = observation[1]
        angle = rc.array([observation[2]])
        v = rc.array([observation[3]])
        omega = rc.array([observation[4]])

        angle_SP = rc.array([self.get_SP_for_PID_angle_arctan(x, y)])

        if self.PID_angle_arctan.SP is None:
            self.PID_angle_arctan.set_SP(angle_SP)

        ANGLE_STABILIZED_TO_ARCTAN = self.PID_angle_arctan.is_stabilized(
            stabilization_tollerance=self.stabilization_tollerance
        )
        XY_STABILIZED_TO_ORIGIN = self.PID_x_y_origin.is_stabilized(
            stabilization_tollerance=self.stabilization_tollerance * 10
        )
        ROBOT_STABILIZED_TO_ORIGIN = self.PID_angle_origin.is_stabilized(
            stabilization_tollerance=self.stabilization_tollerance
        )

        if not ANGLE_STABILIZED_TO_ARCTAN and not np.allclose(
            [x, y], [0, 0], atol=1e-02
        ):

            self.PID_angle_arctan.update_observation_buffer(angle)
            self.PID_angle_origin.reset()
            self.PID_x_y_origin.reset()

            if abs(v) > 1e-2:
                error_derivative = self.current_F / self.m

                F = self.PID_v_zero.compute_action(v, error_derivative=error_derivative)
                M = 0

            else:
                error_derivative = omega

                F = 0
                M = self.PID_angle_arctan.compute_action(
                    angle, error_derivative=error_derivative
                )

        elif ANGLE_STABILIZED_TO_ARCTAN and not XY_STABILIZED_TO_ORIGIN:

            self.PID_x_y_origin.update_observation_buffer(rc.array([x, y]))
            self.PID_angle_arctan.update_observation_buffer(angle)

            self.PID_angle_arctan.reset()
            self.PID_angle_origin.reset()

            # print(f"Stabilize (x, y) to (0, 0), (x, y) = {(x, y)}")

            error_derivative = (
                v * (x * rc.cos(angle) + y * rc.sin(angle)) / rc.sqrt(x ** 2 + y ** 2)
            ) * rc.sign(rc.dot(self.PID_x_y_origin.initial_point, [x, y]))

            F = self.PID_x_y_origin.compute_action(
                [x, y], error_derivative=error_derivative
            )
            self.PID_angle_arctan.set_SP(angle_SP)
            M = self.PID_angle_arctan.compute_action(angle, error_derivative=omega)[0]

        elif XY_STABILIZED_TO_ORIGIN and not ROBOT_STABILIZED_TO_ORIGIN:

            # print("Stabilize angle to 0")

            self.PID_angle_origin.update_observation_buffer(angle)
            self.PID_angle_arctan.reset()
            self.PID_x_y_origin.reset()

            error_derivative = omega

            F = 0
            M = self.PID_angle_origin.compute_action(
                angle, error_derivative=error_derivative
            )

        else:
            self.PID_angle_origin.reset()
            self.PID_angle_arctan.reset()
            self.PID_x_y_origin.reset()

            if abs(v) > 1e-3:
                error_derivative = self.current_F / self.m

                F = self.PID_v_zero.compute_action(v, error_derivative=error_derivative)
                M = 0
            else:
                M = 0
                F = 0

        clipped_F = np.clip(F, -300.0, 300.0)
        clipped_M = np.clip(M, -100.0, 100.0)

        self.current_F = clipped_F
        self.current_M = clipped_M

        return rc.array([np.squeeze(clipped_F), np.squeeze(clipped_M)])

    def compute_action_sampled(self, time, observation):
        """
        Compute sampled action.

        """

        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update internal clock
            self.controller_clock = time

            action = self.compute_action(observation)
            self.times.append(time)

            if self.action_bounds != []:
                for k in range(2):
                    action[k] = np.clip(
                        action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
                    )

            self.action_old = action

            print(action)
            return action

        else:
            return self.action_old

    def reset_all_PID_controllers(self):
        self.PID_x_y_origin.reset()
        self.PID_x_y_origin.reset_buffer()
        self.PID_angle_arctan.reset()
        self.PID_angle_arctan.reset_buffer()
        self.PID_angle_origin.reset()
        self.PID_angle_origin.reset_buffer()

    def reset(self, time_start=0):

        self.controller_clock = time_start

    def compute_LF(self, observation):
        pass


class Controller3WRobotNIMotionPrimitive:
    def __init__(self, K, time_start=0, sampling_time=0.01, action_bounds=None):
        if action_bounds is None:
            action_bounds = []

        self.action_bounds = action_bounds
        self.K = K
        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.Ls = []
        self.times = []
        self.action_old = rc.zeros(2)
        self.clock = Clock(period=sampling_time, time_start=time_start)

    def compute_action(self, observation):
        x = observation[0]
        y = observation[1]
        angle = observation[2]

        angle_cond = np.arctan2(y, x)

        if not np.allclose((x, y), (0, 0), atol=1e-03) and not np.isclose(
            angle, angle_cond, atol=1e-03
        ):
            omega = (
                -self.K
                * np.sign(angle - angle_cond)
                * rc.sqrt(rc.abs(angle - angle_cond))
            )
            v = 0
        elif not np.allclose((x, y), (0, 0), atol=1e-03) and np.isclose(
            angle, angle_cond, atol=1e-03
        ):
            print("cond 2")
            omega = 0
            v = -self.K * rc.sqrt(rc.norm_2(rc.array([x, y])))
        elif np.allclose((x, y), (0, 0), atol=1e-03) and not np.isclose(
            angle, 0, atol=1e-03
        ):
            print("cond 3")
            omega = -self.K * np.sign(angle) * rc.sqrt(rc.abs(angle))
            v = 0
        else:
            omega = 0
            v = 0

        return rc.array([np.clip(v, -25.0, 25.0), np.clip(omega, -5.0, 5.0)])

    def compute_action_sampled(self, time, observation):
        """
        Compute sampled action.

        """

        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update internal clock
            self.controller_clock = time

            action = self.compute_action(observation)
            self.times.append(time)

            if self.action_bounds != []:
                for k in range(2):
                    action[k] = np.clip(
                        action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
                    )

            self.action_old = action
            print(action)
            return action

        else:
            return self.action_old

    def reset(self, time_start=0):
        self.controller_clock = time_start

    def compute_LF(self, observation):
        pass


class Controller3WRobotNIDisassembledCLF:
    """
    Nominal parking controller for NI using disassembled supper_bound_constraintradients.

    """

    def __init__(
        self, controller_gain=10, action_bounds=None, time_start=0, sampling_time=0.1
    ):

        self.controller_gain = controller_gain
        self.action_bounds = action_bounds
        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.Ls = []
        self.times = []
        self.action_old = rc.zeros(2)
        self.clock = Clock(period=sampling_time, time_start=time_start)

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

        return F + 1 / 2 * rc.dot(z, z)

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

        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample

            action = self.compute_action(observation)
            self.times.append(time)

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
        Same as :func:`~Controller3WRobotNIDisassembledCLF.compute_action`, but without invoking the internal clock.

        """

        xNI = self._Cart2NH(observation)
        kappa_val = self._kappa(xNI)
        uNI = self.controller_gain * kappa_val
        action = self._NH2ctrl_Cart(xNI, uNI)

        self.action_old = action
        self.compute_LF(observation)

        return action

    def compute_LF(self, observation):

        xNI = self._Cart2NH(observation)

        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(rc.abs(xNI[2]))
        LF_value = xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma ** 2

        self.Ls.append(LF_value)

        return LF_value


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
