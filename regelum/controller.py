"""Contains high-level structures of controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from abc import ABC, abstractmethod

import numpy as np
import scipy as setpoint
from scipy.optimize import minimize

from .__utilities import rc, Clock
from regelum import RegelumBase
from .policy import Policy
from .critic import Critic, CriticCALF
from typing import Optional, Union
from .objective import RunningObjective
from .data_buffers import DataBuffer


def apply_action_bounds(method):
    def wrapper(self, *args, **kwargs):
        action = method(self, *args, **kwargs)
        if action is not None:
            self.action = action

        if hasattr(self, "action_bounds") and len(self.action_bounds) > 0:
            action = np.clip(
                self.action, self.action_bounds[:, 0], self.action_bounds[:, 1]
            )
            self.action = action
        return self.action

    return wrapper


class Controller(RegelumBase, ABC):
    """A blueprint of optimal controllers."""

    def __init__(
        self,
        policy: Policy,
        time_start: float = 0,
        sampling_time: float = 0.1,
    ):
        """Initialize an instance of Controller.

        :param time_start: time at which simulation started
        :param sampling_time: time interval between two consecutive actions
        """
        super().__init__()
        self.policy = policy
        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.action_old = None

    @apply_action_bounds
    def compute_action_sampled(self, time, state, observation):
        self.is_time_for_new_sample = self.clock.check_time(time)
        if self.is_time_for_new_sample:
            action = self.compute_action(
                time=time,
                state=state,
                observation=observation,
            )
            self.action_old = action
        else:
            action = self.action_old

        return action

    def substitute_constraint_parameters(self, **kwargs):
        self.policy.substitute_parameters(**kwargs)

    @abstractmethod
    @apply_callbacks()
    def compute_action(self, time, state, observation):
        pass

    def optimize_on_event(self, event):
        pass


class RLController(Controller):
    """Controller for policy and value iteration updates."""

    def __init__(
        self,
        policy: Policy,
        critic: Critic,
        running_objective: RunningObjective,
        critic_optimization_event: str,
        policy_optimization_event: str,
        data_buffer_nullify_event: str,
        discount_factor: float = 1.0,
        is_critic_first: bool = False,
        action_bounds: Union[list, np.ndarray, None] = None,
        max_data_buffer_size: Optional[int] = None,
        time_start: float = 0,
        sampling_time: float = 0.1,
    ):
        """Instantiate a RLController object.

        :param policy: Policy object
        :type policy: Policy
        :param critic: Cricit
        :type critic: Critic
        :param running_objective: RunningObjective object
        :type running_objective: RunningObjective
        :param critic_optimization_event: moments when to optimize critic. Can be either 'compute_action' for online learning, or 'reset_episode' for optimizing after each episode, or 'reset_iteration' for optimizing after each iteration
        :type critic_optimization_event: str
        :param policy_optimization_event: moments when to optimize critic. Can be either 'compute_action' for online learning, or 'reset_episode' for optimizing after each episode, or 'reset_iteration' for optimizing after each iteration
        :type policy_optimization_event: str
        :param data_buffer_nullify_event: moments for DataBuffer nullifying. Can be either 'compute_action' for online learning, or 'reset_episode' for optimizing after each episode, or 'reset_iteration' for optimizing after each iteration
        :type data_buffer_nullify_event: str
        :param discount_factor: Discount factor. Used for computing total objective as discounted sum (or integral) of running objectives, defaults to 1.0
        :type discount_factor: float, optional
        :param is_critic_first: if is True then critic is optimized first then policy (can be usefull in DQN or Predictive Algorithms such as RPO, RQL, SQL). For `False` firstly is policy optimized then critic. defaults to False
        :type is_critic_first: bool, optional
        :param action_bounds: action bounds. Applied for every generated action as clip, defaults to None
        :type action_bounds: Union[list, np.ndarray, None], optional
        :param max_data_buffer_size: max size of DataBuffer, if is `None` the DataBuffer is unlimited. defaults to None
        :type max_data_buffer_size: Optional[int], optional
        :param time_start: time at which simulation started, defaults to 0
        :type time_start: float, optional
        :param sampling_time: time interval between two consecutive actions, defaults to 0.1
        :type sampling_time: float, optional
        """
        Controller.__init__(
            self, time_start=time_start, sampling_time=sampling_time, policy=policy
        )

        self.critic_optimization_event = critic_optimization_event
        self.policy_optimization_event = policy_optimization_event
        self.data_buffer_nullify_event = data_buffer_nullify_event
        self.data_buffer = DataBuffer(max_data_buffer_size)
        self.running_objective = running_objective
        self.discount_factor = discount_factor
        self.iteration_counter: int = 0
        self.episode_counter: int = 0
        self.step_counter: int = 0
        self.total_objective: float = 0.0
        self.critic = critic
        self.action_bounds = action_bounds
        self.is_first_compute_action_call = True
        self.is_critic_first = is_critic_first

    def policy_update(self, observation, event, time):
        if self.policy_optimization_event == event:
            self.call_optimize_on_event(self.policy, "compute_action", time)
        if event == "compute_action":
            self.policy.update_action(observation)
            self.update_data_buffer_with_action_stats(observation)

    def update_data_buffer_with_action_stats(self, observation):
        running_objective = self.running_objective(observation, self.policy.action)
        self.data_buffer.push_to_end(
            action=self.policy.action,
            running_objective=running_objective,
            current_total_objective=self.calculate_total_objective(
                running_objective, is_to_update=False
            ),
            observation_action=np.concatenate(
                (observation, self.policy.action), axis=1
            ),
        )

    @apply_callbacks()
    def pre_optimize(
        self,
        which: str,
        event: str,
        time: Optional[int] = None,
    ):
        return which, event, time, self.episode_counter, self.iteration_counter

    def call_optimize_on_event(
        self,
        optimizable_object: Union[Critic, Policy],
        event: str,
        time: Optional[float] = None,
    ):
        if isinstance(optimizable_object, Critic):
            which = "Critic"
        elif isinstance(optimizable_object, Policy):
            which = "Policy"
        else:
            raise ValueError("optimizable object can be either Critic or Policy")

        self.pre_optimize(which=which, event=event, time=time)
        optimizable_object.optimize_on_event(self.data_buffer)

    def critic_update(self, time=None):
        if self.critic_optimization_event == "compute_action":
            self.call_optimize_on_event(self.critic, "compute_action", time)

    @apply_action_bounds
    @apply_callbacks()
    def compute_action(
        self,
        state,
        observation,
        time=0,
    ):
        self.critic.receive_state(state)
        self.policy.receive_state(state)
        self.policy.receive_observation(observation)

        self.data_buffer.push_to_end(
            observation=observation,
            timestamp=time,
            episode_id=self.episode_counter,
            iteration_id=self.iteration_counter,
            step_id=self.step_counter,
        )

        if self.is_first_compute_action_call:
            self.policy_update(observation, "compute_action", time)
            self.is_first_compute_action_call = False
            self.step_counter += 1
            return self.policy.action

        if self.is_critic_first:
            self.critic_update(time)
            self.policy_update(observation, "compute_action", time)
        else:
            self.policy_update(observation, "compute_action", time)
            self.critic_update(time)

        self.step_counter += 1
        return self.policy.action

    def compute_action_sampled(self, time, state, observation):
        action = super().compute_action_sampled(time, state, observation)
        self.calculate_total_objective(
            self.running_objective(observation, action), is_to_update=True
        )
        return action

    def calculate_total_objective(self, running_objective: float, is_to_update=True):
        total_objective = (
            self.total_objective
            + running_objective
            * self.discount_factor**self.clock.current_time
            * self.clock.delta_time
        )
        if is_to_update:
            self.total_objective = total_objective
        else:
            return total_objective

    def optimize_on_event(self, event):
        if event == "reset_iteration":
            self.iteration_counter += 1
            self.episode_counter = 0
            self.step_counter = 0
            self.is_first_compute_action_call = True
        if event == "reset_episode":
            self.episode_counter += 1
            self.step_counter = 0
            self.is_first_compute_action_call = True

        if self.is_critic_first:
            if event == self.critic_optimization_event:
                self.call_optimize_on_event(self.critic, event)
            if event == self.policy_optimization_event:
                self.call_optimize_on_event(self.policy, event)
        else:
            if event == self.policy_optimization_event:
                self.call_optimize_on_event(self.policy, event)
            if event == self.critic_optimization_event:
                self.call_optimize_on_event(self.critic, event)

        if event == self.data_buffer_nullify_event:
            self.data_buffer.nullify_buffer()

    def reset(self):
        """Reset agent for use in multi-episode simulation.

        Only __internal clock and current actions are reset.
        All the learned parameters are retained.

        """
        self.clock.reset()
        self.total_objective = 0.0
        self.policy.action_old = self.policy.action_init
        self.policy.action = self.policy.action_init
        self.is_first_compute_action_call = True


class CALFControllerExPost(RLController):
    """Controller for CALF algorithm."""

    def __init__(
        self,
        policy: Policy,
        critic: CriticCALF,
        safe_controller: Controller,
        running_objective,
        critic_optimization_event: str,
        policy_optimization_event: str,
        data_buffer_nullify_event: str,
        discount_factor=1,
        action_bounds=None,
        max_data_buffer_size: Optional[int] = None,
        time_start: float = 0,
        sampling_time: float = 0.1,
    ):
        """Instantiate a CALFControllerExPost object. The docstring will be completed in the next release.

        :param policy: Pol
        :type policy: Policy
        :param critic: _description_
        :type critic: Critic
        :param safe_controller: _description_
        :type safe_controller: Controller
        :param running_objective: _description_
        :type running_objective: _type_
        :param critic_optimization_event: _description_
        :type critic_optimization_event: str
        :param policy_optimization_event: _description_
        :type policy_optimization_event: str
        :param data_buffer_nullify_event: _description_
        :type data_buffer_nullify_event: str
        :param discount_factor: _description_, defaults to 1
        :type discount_factor: int, optional
        :param action_bounds: _description_, defaults to None
        :type action_bounds: _type_, optional
        :param max_data_buffer_size: _description_, defaults to None
        :type max_data_buffer_size: Optional[int], optional
        :param time_start: _description_, defaults to 0
        :type time_start: float, optional
        :param sampling_time: _description_, defaults to 0.1
        :type sampling_time: float, optional
        """
        super().__init__(
            policy=policy,
            critic=critic,
            running_objective=running_objective,
            critic_optimization_event=critic_optimization_event,
            policy_optimization_event=policy_optimization_event,
            data_buffer_nullify_event=data_buffer_nullify_event,
            discount_factor=discount_factor,
            is_critic_first=True,
            action_bounds=action_bounds,
            max_data_buffer_size=max_data_buffer_size,
            time_start=time_start,
            sampling_time=sampling_time,
        )
        self.safe_controller = safe_controller

    def invoke_safe_action(self, state, observation):
        # self.policy.restore_weights()
        self.critic.restore_weights()
        action = self.safe_controller.compute_action(state, observation)
        self.policy.set_action(action)
        self.update_data_buffer_with_action_stats(observation)

    def update_data_buffer_with_action_stats(self, observation):
        super().update_data_buffer_with_action_stats(observation)
        self.data_buffer.push_to_end(
            observation_last_good=self.critic.observation_last_good
        )

    @apply_callbacks()
    @apply_action_bounds
    def compute_action(
        self,
        state,
        observation,
        time=0,
    ):
        self.critic.receive_state(state)
        self.policy.receive_state(state)
        self.policy.receive_observation(observation)

        self.data_buffer.push_to_end(
            observation=observation,
            timestamp=time,
            episode_id=self.episode_counter,
            iteration_id=self.iteration_counter,
            step_id=self.step_counter,
        )

        if self.is_first_compute_action_call:
            self.critic.observation_last_good = observation
            self.invoke_safe_action(state, observation)
            self.is_first_compute_action_call = False
            self.step_counter += 1
            return self.policy.action

        if rc.norm_2(observation) > 0.5:
            critic_weights = self.critic.optimize_on_event(
                self.data_buffer, is_update_and_cache_weights=False
            )
            critic_weights_accepted = self.critic.opt_status == "success"
        else:
            critic_weights = self.critic.model.weights
            critic_weights_accepted = False

        if critic_weights_accepted:
            self.critic.update_weights(critic_weights)
            self.policy.optimize_on_event(self.data_buffer)
            policy_weights_accepted = True  # self.policy.opt_status == "success"
            if policy_weights_accepted:
                self.policy.update_action(observation)
                self.critic.observation_last_good = observation
                self.update_data_buffer_with_action_stats(observation)
                self.critic.cache_weights(critic_weights)
            else:
                self.invoke_safe_action(state, observation)
        else:
            self.invoke_safe_action(state, observation)

        self.step_counter += 1
        return self.policy.action


class CALFControllerExPostExperimental(RLController):
    """Controller for CALF algorithm."""

    def __init__(
        self,
        policy: Policy,
        critic: CriticCALF,
        safe_controller: Controller,
        running_objective,
        critic_optimization_event: str,
        policy_optimization_event: str,
        data_buffer_nullify_event: str,
        discount_factor=1,
        action_bounds=None,
        max_data_buffer_size: Optional[int] = None,
        time_start: float = 0,
        sampling_time: float = 0.1,
    ):
        """Instantiate a CALFControllerExPost object. The docstring will be completed in the next release.

        :param policy: Pol
        :type policy: Policy
        :param critic: _description_
        :type critic: Critic
        :param safe_controller: _description_
        :type safe_controller: Controller
        :param running_objective: _description_
        :type running_objective: _type_
        :param critic_optimization_event: _description_
        :type critic_optimization_event: str
        :param policy_optimization_event: _description_
        :type policy_optimization_event: str
        :param data_buffer_nullify_event: _description_
        :type data_buffer_nullify_event: str
        :param discount_factor: _description_, defaults to 1
        :type discount_factor: int, optional
        :param action_bounds: _description_, defaults to None
        :type action_bounds: _type_, optional
        :param max_data_buffer_size: _description_, defaults to None
        :type max_data_buffer_size: Optional[int], optional
        :param time_start: _description_, defaults to 0
        :type time_start: float, optional
        :param sampling_time: _description_, defaults to 0.1
        :type sampling_time: float, optional
        """
        super().__init__(
            policy=policy,
            critic=critic,
            running_objective=running_objective,
            critic_optimization_event=critic_optimization_event,
            policy_optimization_event=policy_optimization_event,
            data_buffer_nullify_event=data_buffer_nullify_event,
            discount_factor=discount_factor,
            is_critic_first=True,
            action_bounds=action_bounds,
            max_data_buffer_size=max_data_buffer_size,
            time_start=time_start,
            sampling_time=sampling_time,
        )
        self.safe_controller = safe_controller

    def update_data_buffer_with_action_stats(self, observation):
        super().update_data_buffer_with_action_stats(observation)
        self.data_buffer.push_to_end(
            observation_last_good=self.critic.observation_last_good
        )

    @apply_action_bounds
    @apply_callbacks()
    def compute_action(
        self,
        state,
        observation,
        time=0,
    ):
        self.critic.receive_state(state)
        self.policy.receive_state(state)
        self.policy.receive_observation(observation)

        self.data_buffer.push_to_end(
            observation=observation,
            timestamp=time,
            episode_id=self.episode_counter,
            iteration_id=self.iteration_counter,
            step_id=self.step_counter,
        )

        # if self.is_first_compute_action_call:
        if self.is_first_compute_action_call:
            self.critic.observation_last_good = observation
            self.update_data_buffer_with_action_stats(observation)
            self.is_first_compute_action_call = False
            return self.policy.action

        if rc.norm_2(observation) > 0.5:
            critic_weights = self.critic.optimize_on_event(
                self.data_buffer,
                is_update_and_cache_weights=False,
                is_constrained=rc.norm_2(observation) > 2.0,
            )
            critic_weights_accepted = self.critic.opt_status == "success"
            # print(critic_weights_accepted)
            if critic_weights_accepted:
                self.critic.observation_last_good = observation
        else:
            critic_weights = self.critic.model.weights
            # critic_weights_accepted = False

        # №if critic_weights_accepted:
        self.critic.update_weights(critic_weights)
        self.policy.optimize_on_event(self.data_buffer)
        self.policy.update_action(observation)
        self.update_data_buffer_with_action_stats(observation)
        self.critic.cache_weights(critic_weights)

        self.step_counter += 1
        return self.policy.action


class CALFControllerExPostLegacy(RLController):
    """CALF controller.

    Implements CALF algorithm without predictive constraints.
    """

    def __init__(self, *args, safe_only=False, **kwargs):
        """Initialize an instance of CALFControllerExPost.

        :param args: positional arguments for RLController
        :param safe_only: when safe_only equals True, evaluates actions from safe policy only. Performs CALF updates otherwise.
        :param kwargs: keyword arguments for RLController
        """
        super().__init__(*args, **kwargs)
        if safe_only:
            self.compute_action = self.policy.safe_controller.compute_action
            self.compute_action_sampled = (
                self.policy.safe_controller.compute_action_sampled
            )
            self.reset = self.policy.safe_controller.reset
        self.safe_only = safe_only

    # TODO: DOCSTRING. RENAME TO HUMAN LANGUAGE. DISPLACEMENT?
    def compute_weights_displacement(self, agent):
        self.weights_difference_norm = rc.norm_2(
            self.critic.model.cache.weights - self.critic.optimized_weights
        )
        self.weights_difference_norms.append(self.weights_difference_norm)

    def invoke_safe_action(self, state, observation):
        # self.policy.restore_weights()
        self.critic.restore_weights()
        action = self.policy.safe_controller.compute_action(None, state, observation)

        self.policy.set_action(action)
        self.policy.model.update_and_cache_weights(action)
        self.critic.r_prev += self.policy.running_objective(observation, action)

    # TODO: DOCSTRING
    @apply_callbacks()
    def compute_action(self, state, observation, time=0):
        # Update data buffers
        self.critic.update_buffers(
            observation, self.policy.action
        )  ### store current action and observation in critic's data buffer
        self.critic.receive_state(state)
        # self.critic.safe_decay_param = 1e-1 * rc.norm_2(observation)
        self.policy.receive_observation(
            observation
        )  ### store current observation in policy
        self.policy.receive_state(state)
        self.critic.optimize_weights(time=time)
        critic_weights_accepted = self.critic.opt_status == "success"

        if critic_weights_accepted:
            self.critic.update_weights()

            # self.invoke_safe_action(observation)

            self.policy.optimize_weights(time=time)
            policy_weights_accepted = (
                self.policy.weights_acceptance_status == "accepted"
            )

            if policy_weights_accepted:
                self.policy.update_and_cache_weights()
                self.policy.update_action()

                self.critic.observation_last_good = observation
                self.critic.cache_weights()
                self.critic.r_prev = self.policy.running_objective(
                    observation, self.policy.action
                )
            else:
                self.invoke_safe_action(observation)
        else:
            self.invoke_safe_action(observation)

        # self.collect_critic_stats(time)
        return self.policy.action

    # TODO: NEED IT?
    def collect_critic_stats(self, time):
        self.critic.stabilizing_constraint_violations.append(
            np.squeeze(self.critic.stabilizing_constraint_violation)
        )
        self.critic.lb_constraint_violations.append(0)
        self.critic.ub_constraint_violations.append(0)
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

        self.critic.CALFs.append(current_CALF)


# TODO: DOCSTRING. CLEANUP: NO COMMENTED OUT CODE! NEED ALL DOCSTRINGS HERE
class CALFControllerPredictive(CALFControllerExPost):
    """Predictive CALF controller.

    Implements CALF algorithm without predictive constraints.
    """

    @apply_callbacks()
    def compute_action(
        self,
        state,
        observation,
        time=0,
    ):
        # Update data buffers
        self.critic.update_buffers(
            observation, self.policy.action
        )  ### store current action and observation in critic's data buffer

        # if on prev step weifhtts were acccepted, then upd last good
        if self.policy.weights_acceptance_status == "accepted":
            self.critic.observation_last_good = observation
            self.critic.weights_acceptance_status = "rejected"
            self.policy.weights_acceptance_status = "rejected"
            if self.critic.CALFs != []:
                self.critic.CALFs[-1] = self.critic(
                    self.critic.observation_last_good,
                    use_stored_weights=True,
                )

        # Store current observation in policy
        self.policy.receive_observation(observation)

        self.critic.optimize_weights(time=time)

        if self.critic.weights_acceptance_status == "accepted":
            self.critic.update_weights()

            self.invoke_safe_action(observation)

            self.policy.optimize_weights(time=time)

            if self.policy.weights_acceptance_status == "accepted":
                self.policy.update_and_cache_weights()
                self.policy.update_action()

                self.critic.cache_weights()
            else:
                self.invoke_safe_action(observation)
        else:
            self.invoke_safe_action(observation)

        return self.policy.action


# TODO: DOCSTRING CLEANUP
class Controller3WRobotDisassembledCLF:
    """Nominal controller for 3-wheel robots used for benchmarking of other controllers.

    The controller is sampled.

    For a 3-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here
    a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_]).

    Attributes
    ----------
    m, moment_of_inertia : : numbers
        Mass and moment of inertia around vertical axis of the robot.
    controller_gain : : number
        Controller gain.
    time_start : : number
        Initial value of the controller's __internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    .. [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    .. [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational asetpointects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(
        self,
        m,
        moment_of_inertia,
        controller_gain=10,
        action_bounds=None,
        time_start=0,
        sampling_time=0.01,
        max_iters=200,
        optimizer_engine="SciPy",
    ):
        """Initialize an instance of Controller3WRobotDisassembledCLF.

        :param m: mass of a robot
        :param moment_of_inertia: inertia of a robot
        :param controller_gain: control input multiplier
        :param action_bounds: upper and lower bounds for action yielded from policy
        :param time_start: time at which computations start
        :param sampling_time: a period between two consecutive actions
        :param max_iters: a maximal number of iterations of optimizer
        :param optimizer_engine: optimizer backend. Can be set either CasADi or SciPy
        """
        self.m = m
        self.moment_of_inertia = moment_of_inertia
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

    def reset(self):
        """Reset controller for use in multi-episode simulation."""
        self.action_old = rc.zeros(2)

    def _zeta(self, xNI, theta):
        """Compute generic, i.e., theta-dependent, supper_bound_constraintradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)."""
        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        nablaF = rc.zeros(3, prototype=theta)

        nablaF[0] = (
            4 * xNI[0] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.cos(theta) / sigma_tilde**3
        )

        nablaF[1] = (
            4 * xNI[1] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.sin(theta) / sigma_tilde**3
        )

        nablaF[2] = (
            (
                3 * xNI[0] * rc.cos(theta)
                + 3 * xNI[1] * rc.sin(theta)
                + 2 * rc.sqrt(rc.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * rc.sign(xNI[2])
            / sigma_tilde**3
        )

        return nablaF

    def _kappa(self, xNI, theta):
        """Stabilizing controller for NI-part."""
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
        """Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation."""
        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + rc.sqrt(rc.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma_tilde**2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * rc.dot(z, z)

    def _minimizer_theta(self, xNI, eta):
        thetaInit = 0

        def objective_lambda(theta):
            return self._Fc(xNI, eta, theta)

        if self.optimizer_engine == "SciPy":
            bnds = setpoint.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)
            options = {"maxiter": 50, "disetpoint": False}
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

        else:
            raise NotImplementedError(
                f"Optimizer engine {self.optimizer_engine} not implemented."
            )

        return theta_val

    def _Cart2NH(self, coords_Cart):
        r"""Transform from Cartesian coordinates to non-holonomic (NH) coordinates.

        See Section VIII.A in [[1]_].

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\dot x_3 = x_2 u_1 - x_1 u_2`.

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
        r"""Get control for Cartesian NI from NH coordinates.

        See Section VIII.A in [[1]_].

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\dot x_3 = x_2 u_1 - x_1 u_2`.

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
        uCart[1] = self.moment_of_inertia * uNI[0]

        return uCart

    def compute_action_sampled(self, state, time, observation):
        """See algorithm description in [[1]_], [[2]_].

        **This algorithm needs full-state measurement of the robot**.

        References
        ----------
        .. [1] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
               via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

        .. [2] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational asetpointects." arXiv preprint arXiv:2006.14013 (2020)

        """
        is_time_for_new_sample = self.clock.check_time(time)

        # TODO: REMOVE NASTY COMMENTED OUT DEBUG CODE
        if is_time_for_new_sample:  # New sample
            # This controller needs full-state measurement
            action = self.compute_action(None, observation)

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

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        """Perform the same computation as :func:`~Controller3WRobotDisassembledCLF.compute_action`, but without invoking the __internal clock."""
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


# TODO: IF NEEDED, THEN DOCUMENT IT. MAKE UNIVERSAL, NOT JUST FOR 3W ROBOT
class ControllerMemoryPID:
    """A base class for PID controller.

    This controller is able to use stored data in order to detect whether system is stabilized or not.
    """

    def __init__(
        self,
        P,
        I,
        D,
        setpoint=None,
        sampling_time=0.01,
        initial_point=(-5, -5),
        buffer_length=30,
    ):
        """Initialize an instance of ControllerMemoryPID.

        :param P: proportional gain
        :param I: integral gain
        :param D: differential gain
        :param setpoint: point using as target turing error evaluation
        :param sampling_time: time interval between two consecutive actions
        :param initial_point: point at which computations has begun
        :param buffer_length: length of stored buffer
        """
        self.P = P
        self.I = I
        self.D = D

        self.setpoint = setpoint
        self.integral = 0.0
        self.error_old = 0.0
        self.sampling_time = sampling_time
        self.clock = Clock(period=sampling_time, time_start=0)
        self.initial_point = initial_point
        if isinstance(initial_point, (float, int)):
            self.observation_size = 1
        else:
            self.observation_size = len(initial_point)

        self.buffer_length = buffer_length
        self.observation_buffer = rc.ones((self.observation_size, buffer_length)) * 1e3

    def compute_error(self, process_variable):
        if isinstance(process_variable, (float, int)):
            error = process_variable - self.setpoint
        else:
            if len(process_variable) == 1:
                error = process_variable - self.setpoint
            else:
                norm = rc.norm_2(self.setpoint - process_variable)
                error = norm * rc.sign(rc.dot(self.initial_point, process_variable))
        return error

    def compute_integral(self, error):
        self.integral += error * self.sampling_time
        return self.integral

    def compute_error_derivative_numerically(self, error):
        error_derivative = (error - self.error_old) / self.sampling_time
        self.error_old = error
        return error_derivative

    def compute_action(
        self,
        process_variable,
        error_derivative=None,
        time=0,
    ):
        error = self.compute_error(process_variable)
        integral = self.compute_integral(error)

        if error_derivative is None:
            error_derivative = self.compute_error_derivative_numerically(error)

        PID_signal = -(self.P * error + self.I * integral + self.D * error_derivative)

        ### DEBUG ==============================
        # print(error, integral, error_derivative)
        ### /DEBUG =============================

        return PID_signal

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def update_observation_buffer(self, observation):
        self.observation_buffer = rc.push_vec(self.observation_buffer, observation)

    def set_initial_point(self, point):
        self.initial_point = point

    def reset(self):
        self.integral = 0.0
        self.error_old = 0.0

    def reset_buffer(self):
        self.observation_buffer = (
            rc.ones((self.observation_size, self.buffer_length)) * 1e3
        )

    def is_stabilized(self, stabilization_tollerance=1e-3):
        is_stabilized = np.allclose(
            self.observation_buffer,
            rc.rep_mat(rc.reshape(self.setpoint, (-1, 1)), 1, self.buffer_length),
            atol=stabilization_tollerance,
        )
        return is_stabilized


# TODO: IF NEEDED, THEN DOCUMENT IT
class Controller3WRobotMemoryPID:
    """PID controller for a 3-wheeled robot.

    Uses ControllerMemoryPID controllers wiring.
    """

    def __init__(
        self,
        state_init,
        params=None,
        time_start=0,
        sampling_time=0.01,
        action_bounds=None,
    ):
        """Initialize an instance of Controller3WRobotMemoryPID.

        :param state_init: state at which simulation starts
        :param params: parameters of a 3-wheeled robot
        :param time_start: time at which computations start
        :param sampling_time: time interval between two consecutive computations
        :param action_bounds: upper and lower bounds for action yielded from policy
        """
        if params is None:
            params = [10, 1]

        self.m, self.I = params
        if action_bounds is None:
            action_bounds = []

        self.action_bounds = action_bounds
        self.state_init = state_init

        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.time_start = time_start

        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.Ls = []
        self.times = []
        self.action_old = rc.zeros(2)
        self.PID_angle_arctan = ControllerMemoryPID(
            35, 0.0, 10, initial_point=self.state_init[2]
        )
        self.PID_v_zero = ControllerMemoryPID(
            35, 0.0, 1.2, initial_point=self.state_init[3], setpoint=0.0
        )
        self.PID_x_y_origin = ControllerMemoryPID(
            35,
            0.0,
            35,
            setpoint=rc.array([0.0, 0.0]),
            initial_point=self.state_init[:2],
            # buffer_length=100,
        )
        self.PID_angle_origin = ControllerMemoryPID(
            30, 0.0, 10, setpoint=0.0, initial_point=self.state_init[2]
        )
        self.stabilization_tollerance = 1e-3
        self.current_F = 0
        self.current_M = 0

    def get_setpoint_for_PID_angle_arctan(self, x, y):
        return np.arctan2(y, x)

    def compute_square_of_norm(self, x, y):
        return rc.sqrt(rc.norm_2(rc.array([x, y])))

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        x = observation[0]
        y = observation[1]
        angle = rc.array([observation[2]])
        v = rc.array([observation[3]])
        omega = rc.array([observation[4]])

        angle_setpoint = rc.array([self.get_setpoint_for_PID_angle_arctan(x, y)])

        if self.PID_angle_arctan.setpoint is None:
            self.PID_angle_arctan.set_setpoint(angle_setpoint)

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
                v * (x * rc.cos(angle) + y * rc.sin(angle)) / rc.sqrt(x**2 + y**2)
            ) * rc.sign(rc.dot(self.PID_x_y_origin.initial_point, [x, y]))

            F = self.PID_x_y_origin.compute_action(
                [x, y], error_derivative=error_derivative
            )
            self.PID_angle_arctan.set_setpoint(angle_setpoint)
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

    def compute_action_sampled(self, state, time, observation):
        """Compute sampled action."""
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update __internal clock
            self.controller_clock = time

            action = self.compute_action(observation)
            self.times.append(time)

            self.action_old = action

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

    def reset(self):
        self.clock.reset()
        self.controller_clock = self.time_start

    def compute_LF(self, observation):
        pass


# TODO: IF NEEDED, THEN DOCUMENT IT.
class Controller3WRobotPID:
    """Nominal stabilzing policy for 3wrobot inertial system."""

    def __init__(
        self,
        state_init,
        params=None,
        time_start=0,
        sampling_time=0.01,
        action_bounds=None,
        PID_arctg_params=(10, 0.0, 3),
        PID_v_zero_params=(35, 0.0, 1.2),
        PID_x_y_origin_params=(35, 0.0, 35),
        PID_angle_origin_params=(30, 0.0, 10),
        v_to_zero_bounds=(0.0, 0.05),
        to_origin_bounds=(0.0, 0.1),
        to_arctan_bounds=(0.01, 0.2),
    ):
        """Initialize Controller3WRobotPID.

        :param state_init: initial state of 3wrobot
        :param params: mass and moment of inertia `(M, I)`
        :type params: tuple
        :param time_start: time start
        :param sampling_time: sampling time
        :param action_bounds: bounds that actions should not exceed `[[lower_bound, upper_bound], ...]`
        :param PID_arctg_params: coefficients for PD controller which sets the direction of robot to origin
        :param PID_v_zero_params: coefficients for PD controller which forces speed to zero as robot moves to origin
        :param PID_x_y_origin_params: coefficients for PD controller which moves robot to origin
        :param PID_angle_origin_params: coefficients for PD controller which sets angle to zero near origin
        :param v_to_zero_bounds: bounds for enabling controller which decelerates
        :param to_origin_bounds: bounds for enabling controller which moves robot to origin
        :param to_arctan_bounds: bounds for enabling controller which direct robot to origin
        """
        if params is None:
            params = [10, 1]

        self.m, self.I = params
        if action_bounds is None:
            action_bounds = []

        self.v_to_zero_bounds = v_to_zero_bounds
        self.to_origin_bounds = to_origin_bounds
        self.to_arctan_bounds = to_arctan_bounds

        self.action_bounds = action_bounds
        self.state_init = state_init

        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.time_start = time_start

        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.Ls = []
        self.times = []
        self.action_old = rc.zeros(2)
        self.PID_angle_arctan = ControllerMemoryPID(
            *PID_arctg_params, initial_point=self.state_init[2]
        )
        self.PID_v_zero = ControllerMemoryPID(
            *PID_v_zero_params, initial_point=self.state_init[3], setpoint=0.0
        )
        self.PID_x_y_origin = ControllerMemoryPID(
            *PID_x_y_origin_params,
            setpoint=rc.array([0.0, 0.0]),
            initial_point=self.state_init[:2],
        )
        self.PID_angle_origin = ControllerMemoryPID(
            *PID_angle_origin_params, setpoint=0.0, initial_point=self.state_init[2]
        )
        self.stabilization_tollerance = 1e-3
        self.current_F = 0
        self.current_M = 0

    def get_setpoint_for_PID_angle_arctan(self, x, y, eps=1e-8):
        if abs(x) < eps and abs(y) < eps:
            return 0.0
        return np.arctan2(y, x)

    @staticmethod
    def cdf_uniform(x, a, b):
        loc = a
        scale = 1 / (b - a)
        return np.clip(scale * (x - loc), 0, 1)

    def F_error_derivative(self, x, y, angle, v, eps=1e-8):
        if abs(x) < eps and abs(y) < eps:
            return rc.array([0.0])
        F_error_derivative = (
            v * (x * rc.cos(angle) + y * rc.sin(angle)) / rc.sqrt(x**2 + y**2)
        ) * rc.sign(rc.dot(self.PID_x_y_origin.initial_point, [x, y]))
        return rc.array([F_error_derivative])

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        x = observation[0]
        y = observation[1]
        angle = rc.array([observation[2]])
        v = rc.array([observation[3]])
        omega = rc.array([observation[4]])
        (F_min, F_max), (M_min, M_max) = self.action_bounds[0], self.action_bounds[1]

        self.PID_x_y_origin.set_initial_point(rc.array([x, y]))
        F_error_derivative = self.F_error_derivative(x, y, angle[0], v[0])
        M_arctan_error_derivative = omega
        F_v_to_zero_error_derivative = self.current_F / self.m

        F = self.PID_x_y_origin.compute_action(
            [x, y], error_derivative=F_error_derivative
        )
        angle_setpoint = rc.array([self.get_setpoint_for_PID_angle_arctan(x, y)])
        self.PID_angle_arctan.set_setpoint(angle_setpoint)
        M_arctan = self.PID_angle_arctan.compute_action(
            angle, error_derivative=M_arctan_error_derivative
        )
        M_zero = self.PID_angle_origin.compute_action(angle, error_derivative=omega[0])
        F_v_to_zero = self.PID_v_zero.compute_action(
            v, error_derivative=F_v_to_zero_error_derivative
        )

        lbd_v_to_zero = self.cdf_uniform(
            rc.norm_2(rc.array([x, y, v[0]])),
            self.v_to_zero_bounds[0],
            self.v_to_zero_bounds[1],
        )
        lbd = self.cdf_uniform(
            rc.norm_2(rc.array([x, y])),
            self.to_origin_bounds[0],
            self.to_origin_bounds[1],
        )
        lbd_arctan = self.cdf_uniform(
            rc.abs(angle[0] - self.PID_angle_arctan.setpoint[0]),
            self.to_arctan_bounds[0],
            self.to_arctan_bounds[1],
        )
        control_to_origin = rc.array(
            [
                (1 - lbd_arctan) * np.clip(F[0], F_min, F_max)
                + lbd_arctan * np.clip(F_v_to_zero[0], F_min, F_max),
                lbd_arctan * np.clip(M_arctan[0], M_min, M_max),
            ]
        )
        control_v_to_zero = rc.array([np.clip(F_v_to_zero[0], F_min, F_max), 0.0])
        control_angle_to_zero = rc.array([0, np.clip(M_zero[0], M_min, M_max)])

        action = (
            lbd * control_to_origin
            + (1 - lbd) * control_v_to_zero
            + control_angle_to_zero * (1 - lbd_v_to_zero)
        )

        self.current_F = action[0]
        self.current_M = action[1]

        return action

    def compute_action_sampled(self, time, state, observation):
        """Compute sampled action."""
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update __internal clock
            self.controller_clock = time

            action = self.compute_action(state, observation)
            self.times.append(time)

            self.action_old = action

            return action

        else:
            return self.action_old

    def reset(self):
        self.clock.reset()
        self.controller_clock = self.time_start

    def compute_LF(self, observation):
        pass


class ControllerCartPolePID:
    """A PID controller for Cartpole system."""

    def __init__(
        self,
        action_bounds,
        time_start: float = 0,
        state_init=None,
        sampling_time: float = 0.01,
        system=None,
        upright_gain=None,
        swingup_gain=10,
        pid_loc_thr=0.35,
        pid_scale_thr=10.0,
        clip_bounds=(-1, 1),
    ):
        """Initialize an instance of ControllerCartPolePID.

        :param action_bounds: upper and lower bounds for action yielded from policy
        :param time_start: time at which computations start
        :param state_init: state at which simulation has begun
        :param sampling_time: time interval between two consecutive actions
        :param system: an instance of Cartpole system
        :param upright_gain: gain for PID responsible for stabilization from the upright pole position
        :param swingup_gain: gain for PID responsible for pole swing up
        :param pid_loc_thr: offset of the pole angle responsible for adjusting a moment at which controller is switching from swing up to upright PID controller
        :param pid_scale_thr: multiplier of the pole angle responsible for adjusting a moment at which controller is switching from swing up to upright PID controller
        :param clip_bounds: bounds for clipping observation before passing into PID controller
        """
        if state_init is None:
            state_init = rc.array([np.pi, 0, 0, 0])
        if upright_gain is None:
            upright_gain = rc.array([1, 1, 1, 1])
        self.action_bounds = action_bounds
        self.state_init = state_init
        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.sampling_time = sampling_time
        self.action = np.array([np.mean(action_bounds)])
        self.m_c, self.m_p, self.g, self.l = (
            system.parameters["m_c"],
            system.parameters["m_p"],
            system.parameters["g"],
            system.parameters["l"],
        )
        self.system = system
        self.upright_gain = upright_gain
        self.swingup_gain = swingup_gain
        self.pid_loc_thr = pid_loc_thr
        self.pid_scale_thr = pid_scale_thr
        self.clip_bounds = clip_bounds

    def reset(self):
        pass

    # TODO: DO YOU NEED THIS COMMENT?
    def compute_action_sampled(self, time, state, observation):
        """Compute sampled action."""
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update __internal clock
            self.controller_clock = time

            action = self.compute_action(state, observation)

            if self.action_bounds != []:
                for k in range(len(self.action_bounds)):
                    action[k] = np.clip(
                        action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
                    )

            self.action_old = action
            return action

        else:
            return self.action

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        theta_observed, x, theta_dot, x_dot = observation

        E_total = (
            self.m_p * self.l**2 * theta_dot**2 / 2
            + self.m_p * self.g * self.l * (rc.cos(theta_observed) - 1)
        )

        lbd = (
            1 - rc.tanh((theta_observed - self.pid_loc_thr) * self.pid_scale_thr)
        ) / 2

        low, high = self.clip_bounds
        x_clipped = rc.clip(x, low, high)
        x_dot_clipped = rc.clip(x_dot, low, high)
        self.action = (1 - lbd) * (
            self.swingup_gain * E_total * rc.sign(rc.cos(theta_observed) * theta_dot)
        ) + lbd * self.upright_gain.T @ rc.array(
            [theta_observed, x_clipped, theta_dot, x_dot_clipped]
        )
        return self.action


class ControllerCartPoleEnergyBased:
    """An energy-based controller for cartpole."""

    def __init__(
        self,
        action_bounds,
        time_start: float = 0,
        state_init=None,
        sampling_time: float = 0.01,
        controller_gain=10,
        system=None,
    ):
        """Initialize an instance of ControllerCartPoleEnergyBased.

        :param action_bounds: upper and lower bounds for action yielded from policy
        :param time_start: time at which computations start
        :param state_init: state at which simulation has begun
        :param sampling_time: time interval between two consecutive actions
        :param controller_gain: controller gain
        :param system: an instance of Cartpole system
        """
        if state_init is None:
            state_init = rc.array([np.pi, 0, 0, 0])
        self.action_bounds = action_bounds
        self.state_init = state_init
        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.sampling_time = sampling_time
        self.action = np.array([np.mean(action_bounds)])
        self.controller_gain = controller_gain
        self.m_c, self.m_p, self.g, self.l = (
            system.parameters["m_c"],
            system.parameters["m_p"],
            system.parameters["g"],
            system.parameters["l"],
        )
        self.system = system

    def compute_action_sampled(self, time, state, observation):
        """Compute sampled action."""
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update __internal clock
            self.controller_clock = time

            action = self.compute_action(state, observation)

            if self.action_bounds != []:
                for k in range(len(self.action_bounds)):
                    action[k] = np.clip(
                        action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
                    )

            self.action_old = action
            return action

        else:
            return self.action

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        theta, _, theta_dot, _ = (
            observation[0],
            observation[1],
            observation[2],
            observation[3],
        )

        self.action = (
            self.m_p * self.g * rc.cos(theta) * rc.sin(theta)
            + self.m_p * self.l * theta_dot * rc.sin(theta)
            - self.controller_gain * theta_dot * rc.cos(theta)
        )
        return self.action

    def reset(self):
        pass


class ControllerCartPoleEnergyBasedAdaptive(ControllerCartPoleEnergyBased):
    """Adaptive energy-based controller with adaptation block."""

    def __init__(self, *args, adaptation_block=None, **kwargs):
        """Initialize an instance of energy-based controller with adaptive block.

        :param args: positional arguments for energy-based controller
        :param adaptation_block: an instance of AdaptationBlock
        :param kwargs: keyword arguments for energy-based controllers
        """
        super().__init__(*args, **kwargs)
        self.adaptation_block = adaptation_block

    @apply_action_bounds
    def compute_action(self, state, observation, **kwargs):
        _, _, _, x_dot = (
            observation[0],
            observation[1],
            observation[2],
            observation[3],
        )

        c = self.adaptation_block.get_parameter_estimation(state)

        self.action = super().compute_action(
            state, observation, **kwargs
        ) - c * x_dot**2 * rc.sign(x_dot)
        return self.action


class ControllerLunarLanderPID:
    """Nominal PID controller for lunar lander."""

    def __init__(
        self,
        action_bounds,
        time_start: float = 0,
        state_init=None,
        sampling_time: float = 0.01,
        PID_angle_parameters=None,
        PID_height_parameters=None,
        PID_x_parameters=None,
    ):
        """Initialize an instance of PID controller for lunar lander.

        :param action_bounds: upper and lower bounds for action yielded from policy
        :param time_start: time at which computations start
        :param state_init: state at which simulation has begun
        :param sampling_time: time interval between two consecutive actions
        :param PID_angle_parameters: parameters for PID controller stabilizing angle of lander
        :param PID_height_parameters: parameters for PID controller stabilizing y-coordinate of lander
        :param PID_x_parameters: parameters for PID controller stabilizing x-coordinate of lander
        """
        if state_init is None:
            state_init = rc.array([np.pi, 0, 0, 0])
        if PID_angle_parameters is None:
            PID_angle_parameters = [1, 0, 0]
        if PID_height_parameters is None:
            PID_height_parameters = [10, 0, 0]
        if PID_x_parameters is None:
            PID_x_parameters = [10, 0, 0]
        self.action_bounds = action_bounds
        self.state_init = state_init
        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.sampling_time = sampling_time
        self.action = np.array([np.mean(action_bounds)])
        self.PID_angle = ControllerMemoryPID(
            *PID_angle_parameters,
            initial_point=rc.array([state_init[2]]),
            setpoint=rc.array([0]),
        )
        self.PID_height = ControllerMemoryPID(
            *PID_height_parameters,
            initial_point=rc.array([state_init[1]]),
            setpoint=rc.array([0]),
        )
        self.PID_x = ControllerMemoryPID(
            *PID_x_parameters,
            initial_point=rc.array([state_init[2]]),
            setpoint=rc.array([0]),
        )
        self.threshold_1 = 0.05
        self.threshold_2 = 1.2
        self.threshold = self.threshold_1

    def reset(self):
        pass

    def compute_action_sampled(self, time, state, observation):
        """Compute sampled action."""
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update __internal clock
            self.controller_clock = time

            action = self.compute_action(state, observation)

            if self.action_bounds != []:
                for k in range(len(self.action_bounds)):
                    action[k] = np.clip(
                        action[k], self.action_bounds[k, 0], self.action_bounds[k, 1]
                    )

            self.action_old = action
            return action

        else:
            return self.action

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        self.action = [0, 0]

        if abs(observation[2]) > self.threshold:
            self.threshold = self.threshold_1
            self.action[0] = self.PID_angle.compute_action(
                [rc.array(observation[2])], error_derivative=observation[5]
            )[0]

        else:
            self.threshold = self.threshold_2
            self.action[0] = self.PID_x.compute_action(
                [rc.array(observation[0])], error_derivative=observation[3]
            )[0]
            self.action[1] = self.PID_height.compute_action(
                [rc.array(observation[1])], error_derivative=observation[4]
            )[0]

        self.action = rc.array(self.action)
        return self.action


# TODO: IF NEEDED, THEN DOCUMENT IT.
class Controller2TankPID:
    """PID controller for double tank system."""

    def __init__(
        self,
        action_bounds,
        params=None,
        time_start: float = 0,
        state_init=None,
        sampling_time: float = 0.01,
        PID_2tank_parameters_x1=(1, 0, 0),
        PID_2tank_parameters_x2=(1, 0, 0),
        swing_up_tol=0.1,
    ):
        """Initialize an instance of Controller2TankPID.

        :param action_bounds: upper and lower bounds for action yielded from policy
        :param params: parameters of double tank system
        :param time_start: time at which computations start
        :param state_init: state at which simulation has begun
        :param sampling_time: time interval between two consecutive actions
        :param PID_2tank_parameters_x1: parameters for PID controller stabilizing first component of system's state
        :param PID_2tank_parameters_x2: parameters for PID controller stabilizing second component of system's state
        :param observation_target: ...
        """
        if state_init is None:
            state_init = rc.array([np.pi, 0, 0, 0])
        self.tau1 = 18.4
        self.tau2 = 24.4
        self.K1 = 1.3
        self.K2 = 1
        self.K3 = 0.2

        if params is None:
            params = [self.tau1, self.tau2, self.K1, self.K2, self.K3]
        else:
            self.tau1, self.tau2, self.K1, self.K2, self.K3 = params

        self.action_bounds = action_bounds
        self.state_init = state_init
        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.sampling_time = sampling_time
        self.action = np.array([np.mean(action_bounds)])
        self.PID_2tank_x1 = ControllerMemoryPID(
            *PID_2tank_parameters_x1,
            initial_point=rc.array([state_init[0]]),
        )
        self.PID_2tank_x2 = ControllerMemoryPID(
            *PID_2tank_parameters_x2,
            initial_point=rc.array([state_init[1]]),
        )

    def reset(self):
        pass

    def compute_action_sampled(self, time, state, observation):
        """Compute sampled action."""
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update __internal clock
            self.controller_clock = time

            action = self.compute_action(state, observation)

            self.action_old = action
            return action

        else:
            return self.action

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        # if rc.abs(observation[0]) > np.pi / 2:
        #     action = self.PID_swingup.compute_action(rc.array(observation[0]))
        # else:
        #     action = self.compute_stabilizing_action(observation)
        error_derivative_x1 = -(
            1 / (self.tau1) * (-observation[0] + self.K1 * self.action[0])
        )
        error_derivative_x2 = (
            -1
            / (self.tau2)
            * (
                -observation[1]
                + self.K2 * observation[0]
                + self.K3 * observation[1] ** 2
            )
        )

        self.action = self.PID_2tank_x1.compute_action(
            [rc.array(observation[0])], error_derivative=error_derivative_x1
        ) + self.PID_2tank_x2.compute_action(
            [rc.array(observation[1])], error_derivative=error_derivative_x2
        )
        return self.action


class Controller3WRobotNIDisassembledCLF:
    """Nominal parking controller for NI using disassembled control Lyapunov function."""

    def __init__(
        self, controller_gain=10, action_bounds=None, time_start=0, sampling_time=0.1
    ):
        """Initialize an instance of disassembled-clf controller.

        :param controller_gain: gain of controller
        :param action_bounds: upper and lower bounds for action yielded from policy
        :param time_start: time at which computations start
        :param sampling_time: time interval between two consecutive actions
        """
        self.controller_gain = controller_gain
        self.action_bounds = action_bounds
        self.controller_clock = time_start
        self.time_start = time_start
        self.sampling_time = sampling_time
        self.Ls = []
        self.times = []
        self.action_old = rc.zeros(2)
        self.clock = Clock(period=sampling_time, time_start=time_start)

    def reset(self):
        """Reset controller for use in multi-episode simulation."""
        self.controller_clock = self.time_start
        self.action_old = rc.zeros(2)

    def _zeta(self, xNI):
        """Analytic disassembled supper_bound_constraintradient, without finding minimizer theta."""
        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(abs(xNI[2]))

        nablaL = rc.zeros(3)

        nablaL[0] = (
            4 * xNI[0] ** 3
            + rc.abs(xNI[2]) ** 3
            / sigma**3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[0]
        )
        nablaL[1] = (
            4 * xNI[1] ** 3
            + rc.abs(xNI[2]) ** 3
            / sigma**3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[1]
        )
        nablaL[2] = 3 * rc.abs(xNI[2]) ** 2 * rc.sign(xNI[2]) + rc.abs(
            xNI[2]
        ) ** 3 / sigma**3 * 1 / np.sqrt(rc.abs(xNI[2])) * rc.sign(xNI[2])

        theta = 0

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        nablaF = rc.zeros(3)

        nablaF[0] = (
            4 * xNI[0] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.cos(theta) / sigma_tilde**3
        )
        nablaF[1] = (
            4 * xNI[1] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.sin(theta) / sigma_tilde**3
        )
        nablaF[2] = (
            (
                3 * xNI[0] * rc.cos(theta)
                + 3 * xNI[1] * rc.sin(theta)
                + 2 * np.sqrt(rc.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * rc.sign(xNI[2])
            / sigma_tilde**3
        )

        if xNI[0] == 0 and xNI[1] == 0:
            return nablaF
        else:
            return nablaL

    def _kappa(self, xNI):
        """Stabilizing controller for NI-part."""
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
        """Marginal function for NI."""
        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma_tilde**2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * rc.dot(z, z)

    def _Cart2NH(self, coords_Cart):
        """Transform from Cartesian coordinates to non-holonomic (NH) coordinates."""
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
        """Get control for Cartesian NI from NH coordinates."""
        uCart = rc.zeros(2)

        uCart[0] = uNI[1] + 1 / 2 * uNI[0] * (xNI[2] + xNI[0] * xNI[1])
        uCart[1] = uNI[0]

        return uCart

    def compute_action_sampled(self, time, state, observation):
        """Compute sampled action."""
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            action = self.compute_action(state, observation)
            self.times.append(time)
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

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        """Perform the same computation as :func:`~Controller3WRobotNIDisassembledCLF.compute_action`, but without invoking the __internal clock."""
        xNI = self._Cart2NH(observation)
        kappa_val = self._kappa(xNI)
        uNI = self.controller_gain * kappa_val
        self.action = self._NH2ctrl_Cart(xNI, uNI)

        self.action_old = self.action
        self.compute_LF(observation)

        return self.action

    def compute_LF(self, observation):
        xNI = self._Cart2NH(observation)

        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(rc.abs(xNI[2]))
        LF_value = xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma**2

        self.Ls.append(LF_value)

        return LF_value


class NominalControllerInvertedPendulum:
    """A nominal controller for inverted pendulum representing a PD controller."""

    def __init__(
        self,
        action_bounds,
        controller_gain,
        time_start: float = 0,
        sampling_time: float = 0.1,
    ):
        """Initialize an instance of nominal PD controller.

        :param action_bounds: upper and lower bounds for action yielded from policy
        :param controller_gain: gain of controller
        :param time_start: time at which computations start
        :param sampling_time: time interval between two consecutive actions
        """
        self.action_bounds = action_bounds
        self.controller_gain = controller_gain
        self.observation = np.array([np.pi, 0])
        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.sampling_time = sampling_time
        self.action = np.array([np.mean(action_bounds)])

    def compute_action_sampled(self, time, state, observation, constraints=()):
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:
            self.action = self.compute_action(state, observation, time=time)

        return self.action

    def __call__(self, observation):
        return self.compute_action(observation)

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        self.observation = observation
        return np.array(
            [
                [
                    -((observation[0, 0]) + 0.1 * (observation[0, 1]))
                    * self.controller_gain
                ]
            ]
        )

    def reset(self):
        self.clock.reset()


class Controller3WRobotNIMotionPrimitive(Controller):
    """Controller for non-inertial three-wheeled robot composed of three PID controllers."""

    def __init__(self, K, time_start=0, sampling_time=0.01, action_bounds=None):
        """Initialize an instance of controller.

        :param K: gain of controller
        :param time_start: time at which computations start
        :param sampling_time: time interval between two consecutive actions
        :param action_bounds: upper and lower bounds for action yielded from policy
        """
        super().__init__(
            sampling_time=sampling_time,
            time_start=time_start,
        )
        if action_bounds is None:
            action_bounds = []

        self.action_bounds = action_bounds
        self.K = K
        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.Ls = []
        self.times = []
        self.time_start = time_start

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        x = observation[0, 0]
        y = observation[0, 1]
        angle = observation[0, 2]

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
            omega = 0
            v = -self.K * rc.sqrt(rc.norm_2(rc.hstack([x, y])))
        elif np.allclose((x, y), (0, 0), atol=1e-03) and not np.isclose(
            angle, 0, atol=1e-03
        ):
            omega = -self.K * np.sign(angle) * rc.sqrt(rc.abs(angle))
            v = 0
        else:
            omega = 0
            v = 0

        return rc.force_row(rc.hstack([v, omega]))


class ControllerKinPoint:
    """A nominal controller stabilizing kinematic point (omni-wheel)."""

    def __init__(self, gain, time_start=0, sampling_time=0.01, action_bounds=None):
        """Initialize an instance of kinematic point nominal controller.

        :param gain: gain of controller
        :param time_start: time at which computations start
        :param sampling_time: time interval between two consecutive actions
        :param action_bounds: upper and lower bounds for action yielded from policy
        """
        if action_bounds is None:
            action_bounds = []

        self.action_bounds = action_bounds
        self.gain = gain
        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.Ls = []
        self.times = []
        self.action_old = rc.zeros(2)
        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.time_start = time_start

    @apply_action_bounds
    def compute_action(self, state, observation, time=0):
        return -self.gain * observation

    def compute_action_sampled(self, time, state, observation):
        """Compute sampled action."""
        is_time_for_new_sample = self.clock.check_time(time)

        if is_time_for_new_sample:  # New sample
            # Update __internal clock
            self.controller_clock = time

            action = self.compute_action(state, observation)
            self.times.append(time)
            self.action_old = action
            return action

        else:
            return self.action_old

    def reset(self):
        self.controller_clock = self.time_start
        self.Ls = []
        self.times = []

    def compute_LF(self, observation):
        pass
