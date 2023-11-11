"""Contains high-level structures of controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np

from .__utilities import rc, Clock, AwaitedParameter
from regelum import RegelumBase
from .policy import Policy
from .critic import Critic, CriticCALF
from typing import Optional, Union
from .objective import RunningObjective
from .data_buffers import DataBuffer
from .event import Event
from . import OptStatus
from .simulator import Simulator
from .constraint_parser import ConstraintParser, ConstraintParserTrivial
from .observer import Observer, ObserverTrivial


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


class Controller(RegelumBase):
    """A blueprint of optimal controllers."""

    def __init__(
        self,
        policy: Policy,
        simulator: Simulator,
        time_start: float = 0,
        sampling_time: float = 0.1,
        action_bounds: Optional[list] = None,
        running_objective: Optional[RunningObjective] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        total_objective_threshold: float = np.inf,
        discount_factor: float = 1.0,
    ):
        """Initialize an instance of Controller.

        :param time_start: time at which simulation started
        :param sampling_time: time interval between two consecutive actions
        """
        super().__init__()
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.simulator = simulator
        self.time = self.simulator.time_start
        self.time_old = 0
        self.delta_time = 0
        self.total_objective: float = 0.0
        self.recent_total_objectives_of_episodes = []
        self.total_objectives_of_episodes = []
        self.total_objective_episodic_means = []
        self.action_bounds = np.array(action_bounds)

        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.current_scenario_status = "episode_continues"
        self.total_objective_threshold = total_objective_threshold
        self.discount_factor = discount_factor
        self.is_episode_ended = False
        self.constraint_parser = (
            ConstraintParserTrivial()
            if constraint_parser is None
            else constraint_parser
        )
        self.observer = observer if observer is not None else ObserverTrivial()

        self.state_init, self.action_init = AwaitedParameter(
            "state_init", awaited_from=self.simulator.__class__.__name__
        ), AwaitedParameter(
            "action_init", awaited_from=self.simulator.__class__.__name__
        )
        self.state = self.state_init
        self.action = self.action_init
        self.observation = AwaitedParameter(
            "observation", awaited_from=self.simulator.system.get_observation.__name__
        )

        self.policy = policy
        self.controller_clock = time_start
        self.sampling_time = sampling_time
        self.clock = Clock(period=sampling_time, time_start=time_start)
        self.iteration_counter: int = 0
        self.episode_counter: int = 0
        self.step_counter: int = 0
        self.action_old = AwaitedParameter(
            "action_old", awaited_from=self.compute_action.__name__
        )
        self.running_objective = (
            running_objective
            if running_objective is not None
            else lambda *args, **kwargs: 0
        )
        self.reset()

    def run(self):
        for iteration_counter in range(self.N_iterations):
            self.iteration_counter = iteration_counter
            for episode_counter in range(self.N_episodes):
                self.episode_counter = episode_counter
                while self.sim_status not in [
                    "episode_ended",
                    "simulation_ended",
                    "iteration_ended",
                ]:
                    self.sim_status = self.step()

                self.reload_pipeline()

    def step(self):
        if isinstance(self.action_init, AwaitedParameter) and isinstance(
            self.state_init, AwaitedParameter
        ):
            (
                self.state_init,
                self.action_init,
            ) = self.simulator.get_init_state_and_action()

        if (
            not self.is_episode_ended
            and self.total_objective <= self.total_objective_threshold
        ):
            (
                self.time,
                self.state,
                self.observation,
                self.simulation_metadata,
            ) = self.simulator.get_sim_step_data()

            self.delta_time = (
                self.time - self.time_old
                if self.time_old is not None and self.time is not None
                else 0
            )
            self.time_old = self.time
            if len(list(self.constraint_parser)) > 0:
                self.constraint_parameters = self.constraint_parser.parse_constraints(
                    simulation_metadata=self.simulation_metadata
                )
                self.substitute_constraint_parameters(**self.constraint_parameters)
            estimated_state = self.observer.get_state_estimation(
                self.time, self.observation, self.action
            )

            self.action = self.compute_action_sampled(
                self.time,
                estimated_state,
                self.observation,
            )
            self.simulator.receive_action(self.action)
            self.is_episode_ended = self.simulator.do_sim_step() == -1
            return "episode_continues"
        else:
            self.reset_episode()
            is_iteration_ended = self.episode_counter >= self.N_episodes

            if is_iteration_ended:
                self.reset_iteration()

                is_simulation_ended = self.iteration_counter >= self.N_iterations

                if is_simulation_ended:
                    self.reset_simulation()
                    return "simulation_ended"
                else:
                    return "iteration_ended"
            else:
                return "episode_ended"

    @apply_callbacks()
    def reset_iteration(self):
        self.episode_counter = 0
        self.iteration_counter += 1
        self.recent_total_objectives_of_episodes = self.total_objectives_of_episodes
        self.total_objectives_of_episodes = []

    def reset_episode(self):
        self.episode_counter += 1
        self.is_episode_ended = False

        return self.total_objective

    def reset_simulation(self):
        self.current_scenario_status = "episode_continues"
        self.iteration_counter = 0
        self.episode_counter = 0

    @apply_callbacks()
    def reload_pipeline(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.action = self.action_init
        self.simulator.reset()
        self.reset()
        self.recent_total_objective = self.total_objective
        self.observation = self.simulator.observation
        self.sim_status = 0
        return self.recent_total_objective

    @apply_callbacks()
    def post_compute_action(self, observation, estimated_state):
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "timestamp": self.time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
            "action": self.policy.action,
            "running_objective": self.current_running_objective,
            "current_total_objective": self.total_objective,
        }

    @apply_action_bounds
    def compute_action_sampled(self, time, estimated_state, observation):
        self.is_time_for_new_sample = self.clock.check_time(time)
        if self.is_time_for_new_sample:
            self.on_observation_received(time, estimated_state, observation)
            action = self.compute_action(
                time=time,
                estimated_state=estimated_state,
                observation=observation,
            )
            self.post_compute_action(observation, estimated_state)
            self.step_counter += 1
            self.action_old = action
        else:
            action = self.action_old
        return action

    def compute_action(self, time, estimated_state, observation):
        self.issue_action(observation)
        return self.policy.action

    def issue_action(self, observation):
        self.policy.update_action(observation)

    def __getattribute__(self, name):
        if name == "issue_action":
            return self._issue_action
        else:
            return object.__getattribute__(self, name)

    def _issue_action(self, observation):
        object.__getattribute__(self, "issue_action")(observation)
        self.on_action_issued(observation)

    def on_action_issued(self, observation):
        self.current_running_objective = self.running_objective(
            observation, self.policy.action
        )
        self.total_objective = self.calculate_total_objective(
            self.current_running_objective, self.time
        )
        observation_action = np.concatenate((observation, self.policy.action), axis=1)
        return {
            "action": self.policy.action,
            "running_objective": self.current_running_objective,
            "current_total_objective": self.total_objective,
            "observation_action": observation_action,
        }

    def on_observation_received(self, time, estimated_state, observation):
        self.time = time
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "timestamp": time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
        }

    def substitute_constraint_parameters(self, **kwargs):
        self.policy.substitute_parameters(**kwargs)

    def calculate_total_objective(self, running_objective: float, time: float):
        total_objective = (
            self.total_objective
            + running_objective * self.discount_factor**time * self.sampling_time
        )
        return total_objective

    def reset(self):
        """Reset agent for use in multi-episode simulation.

        Only __internal clock and current actions are reset.
        All the learned parameters are retained.

        """
        self.clock.reset()
        self.total_objective = 0.0
        self.is_first_compute_action_call = True


class RLController(Controller):
    """Controller for policy and value iteration updates."""

    def __init__(
        self,
        policy: Policy,
        critic: Critic,
        running_objective: RunningObjective,
        simulator: Simulator,
        policy_optimization_event: Event,
        data_buffer_nullify_event: Event,
        critic_optimization_event: Event = None,
        discount_factor: float = 1.0,
        is_critic_first: bool = False,
        action_bounds: Optional[Union[list, np.ndarray]] = None,
        max_data_buffer_size: Optional[int] = None,
        time_start: float = 0,
        sampling_time: float = 0.1,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        total_objective_threshold: float = np.inf,
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
            self,
            simulator=simulator,
            time_start=time_start,
            sampling_time=sampling_time,
            policy=policy,
            running_objective=running_objective,
            constraint_parser=constraint_parser,
            observer=observer,
            N_episodes=N_episodes,
            N_iterations=N_iterations,
            total_objective_threshold=total_objective_threshold,
            discount_factor=discount_factor,
            action_bounds=action_bounds,
        )

        self.critic_optimization_event = critic_optimization_event
        self.policy_optimization_event = policy_optimization_event
        self.data_buffer_nullify_event = data_buffer_nullify_event
        self.data_buffer = DataBuffer(max_data_buffer_size)
        self.critic = critic
        self.is_first_compute_action_call = True
        self.is_critic_first = is_critic_first

    def _reset_whatever(self, suffix):
        def wrapped(*args, **kwargs):
            res = object.__getattribute__(self, f"reset_{suffix}")(*args, **kwargs)
            if self.sim_status != "simulation_ended":
                self.optimize(event=getattr(Event, f"reset_{suffix}"))
            return res

        return wrapped

    def __getattribute__(self, name):
        if name == "issue_action":
            return self._issue_action
        elif name.startswith("reset_"):
            suffix = name.split("_")[-1]
            return self._reset_whatever(suffix)
        else:
            return object.__getattribute__(self, name)

    def reload_pipeline(self):
        res = super().reload_pipeline()
        self.critic.reset()
        return res

    def on_observation_received(self, time, estimated_state, observation):
        self.critic.receive_estimated_state(estimated_state)
        self.policy.receive_estimated_state(estimated_state)
        self.policy.receive_observation(observation)

        received_data = super().on_observation_received(
            time, estimated_state, observation
        )
        self.data_buffer.push_to_end(**received_data)

        return received_data

    def on_action_issued(self, observation):
        received_data = super().on_action_issued(observation)
        self.data_buffer.push_to_end(**received_data)
        return received_data

    @apply_action_bounds
    @apply_callbacks()
    def compute_action(
        self,
        estimated_state,
        observation,
        time=0,
    ):
        # Check data consistency
        assert np.allclose(
            estimated_state, self.data_buffer.get_latest("estimated_state")
        )
        assert np.allclose(observation, self.data_buffer.get_latest("observation"))
        assert np.allclose(time, self.data_buffer.get_latest("timestamp"))

        self.optimize(Event.compute_action)
        return self.policy.action

    def optimize(self, event):
        if self.is_first_compute_action_call and event == Event.compute_action:
            self.optimize_or_skip_on_event(self.policy, event)
            self.issue_action(self.data_buffer.get_latest("observation"))
            self.is_first_compute_action_call = False
        elif self.is_critic_first:
            self.optimize_or_skip_on_event(self.critic, event)
            self.optimize_or_skip_on_event(self.policy, event)
            if event == Event.compute_action:
                self.issue_action(self.data_buffer.get_latest("observation"))
        else:
            self.optimize_or_skip_on_event(self.policy, event)
            if event == Event.compute_action:
                self.issue_action(self.data_buffer.get_latest("observation"))
            self.optimize_or_skip_on_event(self.critic, event)

        if event == self.data_buffer_nullify_event:
            self.data_buffer.nullify_buffer()

    def optimize_or_skip_on_event(
        self,
        optimizable_object: Union[Critic, Policy],
        event: str,
    ):
        if (
            isinstance(optimizable_object, Critic)
            and event == self.critic_optimization_event
        ) or (
            isinstance(optimizable_object, Policy)
            and event == self.policy_optimization_event
        ):
            self.pre_optimize(
                optimizable_object=optimizable_object,
                event=event,
                time=self.data_buffer.get_latest("timestamp"),
            )
            optimizable_object.optimize(self.data_buffer)

    @apply_callbacks()
    def pre_optimize(
        self,
        optimizable_object: Union[Critic, Policy],
        event: str,
        time: Optional[int] = None,
    ):
        if isinstance(optimizable_object, Critic):
            which = "Critic"
        elif isinstance(optimizable_object, Policy):
            which = "Policy"
        else:
            raise ValueError("optimizable object can be either Critic or Policy")

        return which, event, time, self.episode_counter, self.iteration_counter


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
        self.policy.restore_weights()
        self.critic.restore_weights()
        action = self.safe_controller.compute_action(state, observation)
        self.policy.set_action(action)
        self.update_data_buffer_with_action_stats(observation)

    def update_data_buffer_with_action_stats(self, observation):
        super().update_data_buffer_with_action_stats(observation)
        self.data_buffer.push_to_end(
            observation_last_good=self.critic.observation_last_good
        )

    def on_observation_received(self, time, estimated_state, observation):
        self.critic.receive_estimated_state(estimated_state)
        self.policy.receive_estimated_state(estimated_state)
        self.policy.receive_observation(observation)

        self.data_buffer.push_to_end(
            observation=observation,
            timestamp=time,
            episode_id=self.episode_counter,
            iteration_id=self.iteration_counter,
            step_id=self.step_counter,
        )

    @apply_callbacks()
    @apply_action_bounds
    def compute_action(
        self,
        state,
        observation,
        time=0,
    ):
        if self.is_first_compute_action_call:
            self.critic.observation_last_good = observation
            self.invoke_safe_action(state, observation)
            self.is_first_compute_action_call = False
            self.step_counter += 1
            return self.policy.action

        if rc.norm_2(observation) > 0.5:
            critic_weights = self.critic.optimize(
                self.data_buffer, is_update_and_cache_weights=False
            )
            critic_weights_accepted = self.critic.opt_status == OptStatus.success
        else:
            critic_weights = self.critic.model.weights
            critic_weights_accepted = False

        if critic_weights_accepted:
            self.critic.update_weights(critic_weights)
            self.policy.optimize(self.data_buffer)
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
