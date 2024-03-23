"""Contains high-level structures of scenarios (agents).

Remarks:

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import torch
import torch.multiprocessing as mp
import random

from regelum.data_buffers import DataBuffer

from regelum.utils import Clock, AwaitedParameter, calculate_value
from regelum import RegelumBase
from .policy import Policy, RLPolicy, PolicyPPO, PolicyReinforce, PolicySDPG, PolicyDDPG
from .critic import Critic, CriticCALF, CriticTrivial
from typing import Optional, Union, Type, Dict, List, Any, Callable
from .objective import RunningObjective
from .data_buffers import DataBuffer
from .event import Event
from . import OptStatus
from .simulator import Simulator
from .constraint_parser import ConstraintParser, ConstraintParserTrivial
from .observer import Observer, ObserverTrivial
from .model import (
    Model,
    ModelNN,
    ModelPerceptron,
    ModelWeightContainer,
    ModelWeightContainerTorch,
    PerceptronWithTruncatedNormalNoise,
    ModelQuadLin,
)
from .predictor import Predictor, EulerPredictor
from regelum.optimizable.core.configs import (
    TorchOptimizerConfig,
    CasadiOptimizerConfig,
)
from regelum.data_buffers.batch_sampler import RollingBatchSampler, EpisodicSampler
from copy import deepcopy
from typing_extensions import Self


class Scenario(RegelumBase):
    """Scenario orchestrator.

    A Scenario orchestrates the training and evaluation cycle of a reinforcement learning agent.
    It runs the simulation based on a given policy, collects observations, applies actions, and
    manages the overall simulation loop, including assessing the agent's performance.
    """

    def __init__(
        self,
        policy: Policy,
        simulator: Simulator,
        sampling_time: float = 0.1,
        running_objective: Optional[RunningObjective] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        value_threshold: float = np.inf,
        discount_factor: float = 1.0,
    ):
        """Initialize the Scenario with the necessary components for running a reinforcement learning experiment.

        Args:
            policy: Policy to generate actions based on observations.
            simulator: Simulator to interact with and collect data for
                training.
            sampling_time: Time interval between action updates.
            running_objective: Objective function for evaluating
                performance.
            constraint_parser: Tool for parsing constraints during
                policy optimization.
            observer: Observer for estimating the system state.
            N_episodes: Total number of episodes to run.
            N_iterations: Total number of iterations to run.
            value_threshold: Threshold to stop the simulation if the
                objective is met.
            discount_factor: Discount factor for future rewards.
        """
        super().__init__()
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.simulator = simulator
        self.time_old = 0
        self.delta_time = 0
        self.value: float = 0.0
        self.recent_values_of_episodes = []

        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.value_threshold = value_threshold
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
        self.sampling_time = sampling_time
        self.clock = Clock(period=sampling_time)
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
        for iteration_counter in range(1, self.N_iterations + 1):
            for episode_counter in range(1, self.N_episodes + 1):
                self.run_episode(
                    episode_counter=episode_counter, iteration_counter=iteration_counter
                )
                self.reload_scenario()

            self.reset_iteration()
            if self.sim_status == "simulation_ended":
                break

    def get_action_from_policy(self):
        return self.simulator.system.apply_action_bounds(self.policy.action)

    def run_episode(self, episode_counter, iteration_counter):
        self.episode_counter = episode_counter
        self.iteration_counter = iteration_counter
        while self.sim_status != "episode_ended":
            self.sim_status = self.step()

    def step(self):
        if isinstance(self.action_init, AwaitedParameter) and isinstance(
            self.state_init, AwaitedParameter
        ):
            (
                self.state_init,
                self.action_init,
            ) = self.simulator.get_init_state_and_action()

        if (not self.is_episode_ended) and (self.value <= self.value_threshold):
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
            return "episode_ended"

    @apply_callbacks()
    def reset_iteration(self):
        pass

    @apply_callbacks()
    def reload_scenario(self):
        self.is_episode_ended = False
        self.recent_value = self.value
        self.observation = self.simulator.observation
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.action = self.action_init
        self.simulator.reset()
        self.reset()
        self.sim_status = 0
        return self.recent_value

    @apply_callbacks()
    def post_compute_action(self, observation, estimated_state):
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "time": self.time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
            "action": self.get_action_from_policy(),
            "running_objective": self.current_running_objective,
            "current_value": self.value,
        }

    def compute_action_sampled(self, time, estimated_state, observation):
        self.is_time_for_new_sample = self.clock.check_time(time)
        if self.is_time_for_new_sample:
            self.on_observation_received(time, estimated_state, observation)
            action = self.simulator.system.apply_action_bounds(
                self.compute_action(
                    time=time,
                    estimated_state=estimated_state,
                    observation=observation,
                )
            )
            self.post_compute_action(observation, estimated_state)
            self.step_counter += 1
            self.action_old = action
        else:
            action = self.action_old
        return action

    def compute_action(self, time, estimated_state, observation):
        self.issue_action(observation)
        return self.get_action_from_policy()

    def issue_action(self, observation):
        self.policy.update_action(observation)

    def __getattribute__(self, name):
        if name == "issue_action":
            return self._issue_action
        else:
            return object.__getattribute__(self, name)

    def _issue_action(self, observation, *args, **kwargs):
        object.__getattribute__(self, "issue_action")(observation, *args, **kwargs)
        self.on_action_issued(observation)

    def on_action_issued(self, observation):
        self.current_running_objective = self.running_objective(
            observation, self.get_action_from_policy()
        )
        self.value = self.calculate_value(self.current_running_objective, self.time)
        observation_action = np.concatenate(
            (observation, self.get_action_from_policy()), axis=1
        )
        return {
            "action": self.get_action_from_policy(),
            "running_objective": self.current_running_objective,
            "current_value": self.value,
            "observation_action": observation_action,
        }

    def on_observation_received(self, time, estimated_state, observation):
        self.time = time
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "time": time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
        }

    def substitute_constraint_parameters(self, **kwargs):
        self.policy.substitute_parameters(**kwargs)

    def calculate_value(self, running_objective: float, time: float):
        value = (
            self.value
            + running_objective * self.discount_factor**time * self.sampling_time
        )
        return value

    def reset(self):
        """Reset agent for use in multi-episode simulation.

        Only __internal clock and current actions are reset.
        All the learned parameters are retained.

        """
        self.clock.reset()
        self.value = 0.0
        self.is_first_compute_action_call = True


class RLScenario(Scenario):
    """Incorporates reinforcement learning algorithms.

    The RLScenario incorporates reinforcement learning algorithms into the Scenario framework,
    enabling iterative optimization of both policies and value functions as part of the agent's learning process.
    """

    def __init__(
        self,
        policy: Policy,
        critic: Critic,
        running_objective: RunningObjective,
        simulator: Simulator,
        policy_optimization_event: Event,
        critic_optimization_event: Event = None,
        discount_factor: float = 1.0,
        is_critic_first: bool = False,
        sampling_time: float = 0.1,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        """Instantiate a RLScenario object.

        Args:
            policy (Policy): Policy object
            critic (Critic): Cricic
            running_objective (RunningObjective): Function to calculate
                the running objective.
            critic_optimization_event (str): moments when to optimize
                critic. Can be either 'compute_action' for online
                learning, or 'reset_episode' for optimizing after each
                episode, or 'reset_iteration' for optimizing after each
                iteration
            policy_optimization_event (str): moments when to optimize
                critic. Can be either 'compute_action' for online
                learning, or 'reset_episode' for optimizing after each
                episode, or 'reset_iteration' for optimizing after each
                iteration
            discount_factor (float, optional): Discount factor. Used for
                computing value as discounted sum (or
                integral) of running objectives, defaults to 1.0
            is_critic_first (bool, optional): if is True then critic is
                optimized first then policy (can be usefull in DQN or
                Predictive Algorithms such as RPV, RQL, SQL). For
                `False` firstly is policy optimized then critic.
                defaults to False
            action_bounds (Union[list, np.ndarray, None], optional):
                action bounds. Applied for every generated action as
                clip, defaults to None
            sampling_time (float, optional): time interval between two
                consecutive actions, defaults to 0.1
        """
        Scenario.__init__(
            self,
            simulator=simulator,
            sampling_time=sampling_time,
            policy=policy,
            running_objective=running_objective,
            constraint_parser=constraint_parser,
            observer=observer,
            N_episodes=N_episodes,
            N_iterations=N_iterations,
            value_threshold=value_threshold,
            discount_factor=discount_factor,
        )
        is_parallel = self._metadata["argv"].parallel

        self.critic_optimization_event = critic_optimization_event
        self.policy_optimization_event = policy_optimization_event
        self.data_buffer = DataBuffer()
        self.critic = critic
        self.is_first_compute_action_call = True
        self.is_critic_first = is_critic_first
        self.stopping_criterion = stopping_criterion
        self.is_parallel = is_parallel
        self.seed_increment = 0
        self.annealing_exploration_factor = 1

    def _reset_whatever(self, suffix):
        def wrapped(*args, **kwargs):
            res = object.__getattribute__(self, f"reset_{suffix}")(*args, **kwargs)
            if self.sim_status != "simulation_ended":
                self.optimize(event=getattr(Event, f"reset_{suffix}"))
            if suffix == "iteration":
                self.data_buffer.nullify_buffer()
            return res

        return wrapped

    def run_episode(self, episode_counter, iteration_counter):
        super().run_episode(
            episode_counter=episode_counter, iteration_counter=iteration_counter
        )
        return self.data_buffer

    def run_ith_scenario(
        self, episode_id: int, iteration_id: int, scenarios: List[Self], queue
    ):
        seed = torch.initial_seed() + episode_id
        torch.manual_seed(seed)
        # np.random.seed(seed)
        random.seed(seed)
        self.seed_increment += 1

        queue.put(
            (
                episode_id,
                scenarios[episode_id - 1].run_episode(episode_id, iteration_id),
            )
        )

    def instantiate_rl_scenarios(self):
        simulators = [deepcopy(self.simulator) for _ in range(self.N_episodes)]
        # for simulator in simulators:
        #     simulator.state_init = simulator.state + np.random.normal(
        #         0, 0.5, simulator.state.shape
        #     )
        #     simulator.state = simulator.state_init
        #     simulator.system.state = simulator.state_init
        #     simulator.observation = simulator.get_observation(
        #         time=simulator.time,
        #         state=simulator.state_init,
        #         inputs=simulator.action_init,
        #     )
        # simulator.system.state_init = simulator.state_init
        scenarios = [
            RLScenario(
                policy=self.policy,
                critic=self.critic,
                running_objective=self.running_objective,
                simulator=simulators[i],
                policy_optimization_event=self.policy_optimization_event,
                critic_optimization_event=self.critic_optimization_event,
                discount_factor=self.discount_factor,
                is_critic_first=self.is_critic_first,
                sampling_time=self.sampling_time,
                constraint_parser=self.constraint_parser,
                observer=self.observer,
                N_episodes=self.N_episodes,  # for correct logging
                N_iterations=self.N_iterations,  # for correct logging
                value_threshold=self.value_threshold,
                stopping_criterion=self.stopping_criterion,
            )
            for i in range(self.N_episodes)
        ]
        return scenarios

    @apply_callbacks()
    def dump_data_buffer(self, episode_id: int, data_buffer: DataBuffer):
        return episode_id, data_buffer

    def run(self):
        if not self.is_parallel:
            return super().run()

        for self.iteration_counter in range(1, self.N_iterations + 1):
            # self.policy.model.stds *= self.annealing_exploration_factor
            one_episode_rl_scenarios = self.instantiate_rl_scenarios()
            result_queue = mp.Queue()
            args = [
                (i, self.iteration_counter, one_episode_rl_scenarios, result_queue)
                for i in range(1, self.N_episodes + 1)
            ]
            processes = [
                mp.Process(target=self.run_ith_scenario, args=arg) for arg in args
            ]
            for p in processes:
                p.start()

            enumerated_data_buffers = sorted(
                [result_queue.get() for _ in range(self.N_episodes)], key=lambda x: x[0]
            )

            for p in processes:
                p.join()

            for (episode_idx, data_buffer), scenario in zip(
                enumerated_data_buffers, one_episode_rl_scenarios
            ):
                self.dump_data_buffer(episode_idx, data_buffer)
                self.data_buffer.concat(data_buffer)
                data = data_buffer.to_pandas(["running_objective", "time"])
                scenario.value = calculate_value(
                    data["running_objective"],
                    data["time"],
                    self.discount_factor,
                    self.sampling_time,
                )
                scenario.reload_scenario()

            self.reset_iteration()
            if self.sim_status == "simulation_ended":
                break

    def reset_iteration(self):
        super().reset_iteration()
        if self.stopping_criterion is not None:
            if self.stopping_criterion(self.data_buffer):
                self.sim_status = "simulation_ended"

    def __getattribute__(self, name):
        if name == "issue_action":
            return self._issue_action
        elif name.startswith("reset_"):
            suffix = name.split("_")[-1]
            return self._reset_whatever(suffix)
        else:
            return object.__getattribute__(self, name)

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
        assert np.allclose(time, self.data_buffer.get_latest("time"))

        self.optimize(Event.compute_action)
        return self.get_action_from_policy()

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
                time=self.data_buffer.get_latest("time"),
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


class CALFScenario(RLScenario):
    def __init__(
        self,
        policy: Policy,
        critic: CriticCALF,
        safe_policy: Policy,
        simulator: Simulator,
        running_objective: RunningObjective,
        discount_factor: float,
        sampling_time: float,
        observer: Optional[Observer] = None,
        N_iterations: int = 5,
        critic_learning_norm_threshold: float = 0.1,
        store_weights_thr: float = 0.0,
        weighted_norm_coeffs: Optional[List[float]] = None,
        weights_disturbance_std_after_iteration: Optional[float] = None,
        is_mean_weighted: bool = False,
        is_safe_filter_on: bool = True,
    ):
        super().__init__(
            policy=policy,
            critic=critic,
            simulator=simulator,
            running_objective=running_objective,
            critic_optimization_event=Event.compute_action,
            policy_optimization_event=Event.compute_action,
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            observer=observer,
            N_iterations=N_iterations,
        )
        self.is_mean_weighted = is_mean_weighted
        self.safe_policy = safe_policy
        self.critic_learning_norm_threshold = critic_learning_norm_threshold
        self.weighted_norm_coeffs = (
            np.array(weighted_norm_coeffs)
            if weighted_norm_coeffs is not None
            else np.ones(self.policy.system.dim_observation)
        )
        self.store_weights_thr = store_weights_thr
        if store_weights_thr > 0.0:
            self.store_weights_for_next_iteration_trigger = (
                lambda state, thr: np.linalg.norm(state * self.weighted_norm_coeffs)
                > thr
            )
        else:
            self.store_weights_for_next_iteration_trigger = lambda state, thr: True

        self.stored_weights = self.critic.weights
        self.weights_disturbance_std_after_iteration = (
            weights_disturbance_std_after_iteration
        )
        self.is_safe_filter_on = is_safe_filter_on

    def issue_action(self, observation, is_safe=False):
        if is_safe:
            self.policy.restore_weights()
            self.critic.restore_weights()
            safe_action = self.safe_policy.get_action(observation)
            self.policy.set_action(safe_action)
        else:
            self.policy.update_action(observation)

    def on_action_issued(self, observation):
        data = super().on_action_issued(observation)
        data |= {"observation_last_good": self.critic.observation_last_good}
        self.data_buffer.push_to_end(
            observation_last_good=self.critic.observation_last_good
        )

        return data

    def compute_action_sampled(self, time, estimated_state, observation):
        self.is_time_for_new_sample = self.clock.check_time(time)
        if self.is_time_for_new_sample:
            self.on_observation_received(time, estimated_state, observation)
            action = self.simulator.system.apply_action_bounds(
                self.compute_action(
                    time=time,
                    estimated_state=estimated_state,
                    observation=observation,
                    is_update_critic=np.linalg.norm(
                        observation * self.weighted_norm_coeffs
                    )
                    > self.critic_learning_norm_threshold,
                )
            )
            self.post_compute_action(observation, estimated_state)
            self.step_counter += 1
            self.action_old = action
        else:
            action = self.action_old
        return action

    @apply_callbacks()
    def compute_action(
        self, estimated_state, observation, time=0, is_update_critic=True
    ):
        assert np.allclose(
            estimated_state, self.data_buffer.get_latest("estimated_state")
        )
        assert np.allclose(observation, self.data_buffer.get_latest("observation"))
        assert np.allclose(time, self.data_buffer.get_latest("time"))

        if self.is_first_compute_action_call:
            self.critic.observation_last_good = observation
            self.weights_for_next_iteration = self.critic.weights
            self.issue_action(observation, is_safe=True)
            self.is_first_compute_action_call = False
        else:
            self.pre_optimize(self.critic, Event.compute_action, time)
            if is_update_critic:
                critic_weights = self.critic.optimize(
                    self.data_buffer, is_update_and_cache_weights=False
                )
                critic_weights_accepted = (
                    self.critic.opt_status == OptStatus.success
                ) or not self.is_safe_filter_on

            else:
                critic_weights_accepted = False
                critic_weights = self.critic.weights
                self.critic.opt_status = OptStatus.failed

            if critic_weights_accepted:
                self.critic.update_weights(critic_weights)
                if self.store_weights_for_next_iteration_trigger(
                    observation, self.store_weights_thr
                ):
                    # print("STORED")
                    self.stored_weights = critic_weights
                self.pre_optimize(self.policy, Event.compute_action, time)
                self.policy.optimize(self.data_buffer)
                policy_weights_accepted = (
                    self.policy.opt_status == OptStatus.success
                ) or not self.is_safe_filter_on

                if policy_weights_accepted:
                    self.critic.observation_last_good = observation
                    self.issue_action(observation, is_safe=False)
                    self.critic.cache_weights(critic_weights)
                else:
                    self.issue_action(observation, is_safe=True)
            else:
                self.issue_action(observation, is_safe=True)

            step_id = self.data_buffer.get_latest("step_id")
            self.weights_for_next_iteration = (
                self.weights_for_next_iteration * (step_id - 1)
                + self.critic.weights * np.exp(-step_id)
            ) / step_id

        return self.get_action_from_policy()

    def reset_iteration(self):
        super().reset_iteration()
        if self.stored_weights is not None and self.store_weights_thr > 0.0:
            self.critic.model.update_and_cache_weights(self.stored_weights)
        self.critic.observation_last_good = self.simulator.observation_init
        # if self.weights_disturbance_std_after_iteration is not None:
        #     (
        #         self.critic.model.update_and_cache_weights(
        #             self.critic.weights
        #             + abs(np.random.randn(self.critic.model.weights.shape[0]))
        #             * self.weights_disturbance_std_after_iteration
        #         )
        #     )
        if self.is_mean_weighted:
            self.critic.model.update_and_cache_weights(self.weights_for_next_iteration)


class CALF(CALFScenario):
    """Scenario for CALF algorithm."""

    def __init__(
        self,
        simulator: Simulator,
        running_objective: RunningObjective,
        safe_policy: Policy,
        critic_td_n: int = 2,
        observer: Optional[Observer] = None,
        prediction_horizon: int = 1,
        policy_model: Optional[Model] = None,
        critic_model: Optional[Model] = None,
        predictor: Optional[Predictor] = None,
        discount_factor=1.0,
        sampling_time: float = 0.1,
        critic_lb_parameter: float = 0.0,
        critic_ub_parameter: float = 1.0,
        critic_safe_decay_param: float = 0.001,
        critic_is_dynamic_decay_rate: bool = False,
        critic_batch_size: int = 10,
        critic_regularization_param: float = 0,
        critic_learning_norm_threshold: float = 0.0,
        N_iterations=5,
        store_weights_thr=0,
        weighted_norm_coeffs: Optional[List[float]] = None,
        weights_disturbance_std_after_iteration: Optional[float] = None,
        is_mean_weighted: bool = False,
        is_safe_filter_on: bool = True,
    ):
        """Instantiate CALF class.

        Args:
            simulator (Simulator): The simulator object.
            running_objective (RunningObjective): The running objective.
            safe_policy (Policy): The safe policy.
            critic_td_n (int, optional): The TD-N parameter for the
                critic. Defaults to 2.
            observer (Optional[Observer], optional): The observer
                object. Defaults to None.
            prediction_horizon (int, optional): The prediction horizon.
                Defaults to 1.
            policy_model (Optional[Model], optional): The policy model.
                Defaults to None.
            critic_model (Optional[Model], optional): The critic model.
                Defaults to None.
            predictor (Optional[Predictor], optional): The predictor
                object. Defaults to None.
            discount_factor (float, optional): The discount factor.
                Defaults to 1.0.
            sampling_time (float, optional): The sampling time. Defaults
                to 0.1.
            critic_lb_parameter (float, optional): The lower bound
                parameter for the critic. Defaults to 0.0.
            critic_ub_parameter (float, optional): The upper bound
                parameter for the critic. Defaults to 1.0.
            critic_safe_decay_param (float, optional): The safe decay
                parameter for the critic. Defaults to 0.001.
            critic_is_dynamic_decay_rate (bool, optional): Whether the
                critic has a dynamic decay rate. Defaults to False.
            critic_batch_size (int, optional): The batch size for the
                critic optimizer. Defaults to 10.

        Returns:
            None: None
        """
        system = simulator.system
        critic = CriticCALF(
            system=system,
            model=critic_model,
            td_n=critic_td_n,
            is_same_critic=False,
            is_value_function=True,
            discount_factor=1.0,
            sampling_time=sampling_time,
            safe_decay_param=critic_safe_decay_param,
            is_dynamic_decay_rate=critic_is_dynamic_decay_rate,
            safe_policy=safe_policy,
            lb_parameter=critic_lb_parameter,
            ub_parameter=critic_ub_parameter,
            optimizer_config=CasadiOptimizerConfig(critic_batch_size),
            regularization_param=critic_regularization_param,
        )
        policy = RLPolicy(
            action_bounds=system.action_bounds,
            model=(
                ModelWeightContainer(
                    weights_init=np.zeros(
                        (prediction_horizon, system.dim_inputs),
                        dtype=np.float64,
                    ),
                    dim_output=system.dim_inputs,
                )
                if policy_model is None
                else policy_model
            ),
            system=system,
            running_objective=running_objective,
            prediction_horizon=prediction_horizon,
            algorithm="rpv",
            critic=critic,
            predictor=(
                predictor
                if predictor is not None
                else EulerPredictor(system=system, pred_step_size=sampling_time)
            ),
            discount_factor=discount_factor,
            optimizer_config=CasadiOptimizerConfig(),
        )

        super().__init__(
            policy=policy,
            critic=critic,
            safe_policy=safe_policy,
            simulator=simulator,
            running_objective=running_objective,
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            observer=observer,
            N_iterations=N_iterations,
            critic_learning_norm_threshold=critic_learning_norm_threshold,
            store_weights_thr=store_weights_thr,
            weighted_norm_coeffs=weighted_norm_coeffs,
            weights_disturbance_std_after_iteration=weights_disturbance_std_after_iteration,
            is_mean_weighted=is_mean_weighted,
            is_safe_filter_on=is_safe_filter_on,
        )


class CALFTorch(CALFScenario):
    def __init__(
        self,
        simulator: Simulator,
        running_objective: RunningObjective,
        safe_policy: Policy,
        policy_n_epochs: int,
        critic_n_epochs: int,
        critic_n_epochs_per_constraint: int,
        policy_opt_method_kwargs: Dict[str, Any],
        critic_opt_method_kwargs: Dict[str, Any],
        policy_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_td_n: int = 2,
        observer: Optional[Observer] = None,
        prediction_horizon: int = 1,
        policy_model: Optional[Model] = None,
        critic_model: Optional[Model] = None,
        predictor: Optional[Predictor] = None,
        discount_factor=1.0,
        sampling_time: float = 0.1,
        critic_lb_parameter: float = 0.0,
        critic_ub_parameter: float = 1.0,
        critic_safe_decay_param: float = 0.001,
        critic_is_dynamic_decay_rate: bool = False,
        critic_batch_size: int = 10,
        N_iterations=5,
    ):
        system = simulator.system
        critic = CriticCALF(
            system=system,
            model=critic_model,
            td_n=critic_td_n,
            is_same_critic=False,
            is_value_function=True,
            discount_factor=1.0,
            sampling_time=sampling_time,
            safe_decay_param=critic_safe_decay_param,
            is_dynamic_decay_rate=critic_is_dynamic_decay_rate,
            safe_policy=safe_policy,
            lb_parameter=critic_lb_parameter,
            ub_parameter=critic_ub_parameter,
            optimizer_config=TorchOptimizerConfig(
                n_epochs=critic_n_epochs,
                opt_method=critic_opt_method,
                opt_method_kwargs=critic_opt_method_kwargs,
                data_buffer_iter_bathes_kwargs=dict(
                    batch_sampler=RollingBatchSampler,
                    dtype=torch.FloatTensor,
                    mode="backward",
                    batch_size=critic_batch_size,
                ),
                n_epochs_per_constraint=critic_n_epochs_per_constraint,
            ),
        )
        policy = RLPolicy(
            action_bounds=system.action_bounds,
            model=(
                ModelWeightContainerTorch(
                    dim_weights=(prediction_horizon, system.dim_inputs),
                    output_bounds=system.action_bounds,
                    output_bounding_type="clip",
                )
                if policy_model is None
                else policy_model
            ),
            system=system,
            running_objective=running_objective,
            prediction_horizon=prediction_horizon,
            algorithm="rpv",
            critic=critic,
            predictor=(
                predictor
                if predictor is not None
                else EulerPredictor(system=system, pred_step_size=sampling_time)
            ),
            discount_factor=discount_factor,
            optimizer_config=TorchOptimizerConfig(
                n_epochs=policy_n_epochs,
                opt_method=policy_opt_method,
                opt_method_kwargs=policy_opt_method_kwargs,
                data_buffer_iter_bathes_kwargs=dict(
                    batch_sampler=RollingBatchSampler,
                    dtype=torch.FloatTensor,
                    mode="backward",
                    batch_size=1,
                    n_batches=1,
                ),
            ),
        )

        super().__init__(
            policy=policy,
            critic=critic,
            safe_policy=safe_policy,
            simulator=simulator,
            running_objective=running_objective,
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            observer=observer,
            N_iterations=N_iterations,
        )


class MPC(RLScenario):
    """Leverages the Model Predictive Control Scenario.

    The MPCScenario leverages the Model Predictive Control (MPC) approach within the reinforcement learning scenario,
    utilizing a prediction model to plan and apply sequences of actions that optimize the desired objectives over a time horizon.
    """

    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        predictor: Optional[Predictor] = None,
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        discount_factor: float = 1.0,
    ):
        """Initialize the MPC agent, setting up the required structures for MPC.

        Args:
            running_objective (RunningObjective): The objective function
                to assess the costs over the prediction horizon.
            simulator (Simulator): The environment simulation for
                applying and testing the agent.
            prediction_horizon (int): The number of steps into the
                future over which predictions are made.
            predictor (Optional[Predictor]): The prediction model used
                for forecasting future states.
            sampling_time (float): The time step interval for scenario.
            observer (Observer | None): The component for estimating the
                system's current state. Defaults to None.
            constraint_parser (Optional[ConstraintParser]): The
                mechanism for enforcing operational constraints.
                Defaults to None.
            discount_factor (float): The factor for discounting the
                value of future costs. Defaults to 1.0.
        """
        system = simulator.system
        super().__init__(
            N_episodes=1,
            N_iterations=1,
            simulator=simulator,
            policy_optimization_event=Event.compute_action,
            critic=CriticTrivial(),
            running_objective=running_objective,
            observer=observer,
            sampling_time=sampling_time,
            policy=RLPolicy(
                action_bounds=system.action_bounds,
                model=ModelWeightContainer(
                    weights_init=np.zeros(
                        (prediction_horizon + 1, system.dim_inputs), dtype=np.float64
                    ),
                    dim_output=system.dim_inputs,
                ),
                constraint_parser=constraint_parser,
                system=system,
                running_objective=running_objective,
                prediction_horizon=prediction_horizon,
                algorithm="mpc",
                critic=CriticTrivial(),
                predictor=(
                    predictor
                    if predictor is not None
                    else EulerPredictor(system=system, pred_step_size=sampling_time)
                ),
                discount_factor=discount_factor,
                optimizer_config=CasadiOptimizerConfig(),
            ),
        )


def get_predictive_kwargs(
    running_objective: RunningObjective,
    simulator: Simulator,
    prediction_horizon: int,
    predictor: Predictor,
    sampling_time: float,
    observer: Optional[Observer],
    constraint_parser: Optional[ConstraintParser],
    N_iterations: int,
    discount_factor: float,
    algorithm_name: str,
    stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    policy_model: Optional[Model] = None,
    critic_td_n: Optional[int] = None,
    critic_is_on_policy: Optional[bool] = None,
    critic_is_value_function: Optional[bool] = None,
    critic_regularization_param: Optional[float] = None,
    critic_is_same_critic: Optional[bool] = None,
    critic_model: Optional[Model] = None,
    critic_n_samples: Optional[int] = None,
    epsilon_random_parameter: Optional[float] = None,
):
    system = simulator.system
    critic = Critic(
        system=system,
        model=(
            critic_model
            if critic_model is not None
            else ModelQuadLin("diagonal", is_with_linear_terms=False)
        ),
        td_n=critic_td_n,
        is_on_policy=critic_is_on_policy,
        is_value_function=critic_is_value_function,
        sampling_time=sampling_time,
        discount_factor=discount_factor,
        regularization_param=critic_regularization_param,
        action_bounds=system.action_bounds,
        is_same_critic=critic_is_same_critic,
        optimizer_config=CasadiOptimizerConfig(batch_size=critic_n_samples),
    )

    policy = RLPolicy(
        action_bounds=system.action_bounds,
        model=(
            ModelWeightContainer(
                weights_init=np.zeros(
                    (
                        prediction_horizon + (1 if algorithm_name != "rpv" else 0),
                        system.dim_inputs,
                    ),
                    dtype=np.float64,
                ),
                dim_output=system.dim_inputs,
            )
            if policy_model is None
            else policy_model
        ),
        constraint_parser=constraint_parser,
        system=system,
        running_objective=running_objective,
        prediction_horizon=prediction_horizon,
        algorithm=algorithm_name,
        critic=critic,
        predictor=(
            predictor
            if predictor is not None
            else EulerPredictor(system=system, pred_step_size=sampling_time)
        ),
        discount_factor=discount_factor,
        optimizer_config=CasadiOptimizerConfig(batch_size=1),
        epsilon_random_parameter=epsilon_random_parameter,
    )

    return dict(
        stopping_criterion=stopping_criterion,
        N_episodes=1,
        N_iterations=N_iterations,
        simulator=simulator,
        policy_optimization_event=Event.compute_action,
        critic_optimization_event=Event.compute_action,
        critic=critic,
        running_objective=running_objective,
        observer=observer,
        sampling_time=sampling_time,
        policy=policy,
        is_critic_first=True,
    )


class SQL(RLScenario):
    """Implements Stacked Q-Learning algorithm."""

    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        critic_td_n: int,
        critic_batch_size: int,
        predictor: Optional[Predictor] = None,
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        N_iterations: int = 1,
        discount_factor: float = 1.0,
        policy_model: Optional[Model] = None,
        critic_model: Optional[Model] = None,
        critic_regularization_param: float = 0.0,
        epsilon_random_parameter: Optional[float] = None,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        """Instantiate SQL.

        Args:
            running_objective (RunningObjective): The objective function
                to assess the costs over the prediction horizon.
            simulator (Simulator): The environment simulation for
                applying and testing the agent.
            prediction_horizon (int): The number of steps into the
                future over which predictions are made.
            critic_td_n (int): The n-step temporal-difference parameter
                used for critic updates.
            critic_batch_size (int): The batch size for the critic.
            predictor (Optional[Predictor]): The prediction model used
                for forecasting future states. Defaults to None.
            sampling_time (float): The time step interval for scenario.
                Defaults to 0.1.
            observer (Optional[Observer]): The observer object used for
                the algorithm. Defaults to None.
            constraint_parser (Optional[ConstraintParser]): The
                component for estimating the system's current state.
                Defaults to None.
            N_iterations (int): The number of iterations for the
                algorithm. Defaults to 1.
            discount_factor (float): The factor for discounting the
                value of future costs. Defaults to 1.0.
            policy_model (Optional[Model]): The model parameterizing the
                policy. Defaults to None. If `None` then
                `ModelWeightContainer` is used.
            critic_model (Optional[Model]): The model parameterizing the
                critic. Defaults to None. If `None` then diagonal
                quadratic form is used.
            critic_regularization_param (float): The regularization
                parameter for the critic. Defaults to 0.

        Returns:
            None: None
        """
        super().__init__(
            **get_predictive_kwargs(
                simulator=simulator,
                running_objective=running_objective,
                prediction_horizon=prediction_horizon,
                predictor=predictor,
                sampling_time=sampling_time,
                observer=observer,
                constraint_parser=constraint_parser,
                N_iterations=N_iterations,
                discount_factor=discount_factor,
                algorithm_name="sql",
                policy_model=policy_model,
                critic_model=critic_model,
                critic_regularization_param=critic_regularization_param,
                critic_td_n=critic_td_n,
                critic_is_on_policy=True,
                critic_is_value_function=False,
                critic_is_same_critic=False,
                critic_n_samples=critic_batch_size,
                epsilon_random_parameter=epsilon_random_parameter,
                stopping_criterion=stopping_criterion,
            )
        )


class RQL(RLScenario):
    """Implements Rollout Q-Learning algorithm."""

    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        critic_td_n: int,
        critic_batch_size: int,
        predictor: Optional[Predictor] = None,
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        N_iterations: int = 1,
        discount_factor: float = 1.0,
        policy_model: Optional[Model] = None,
        critic_model: Optional[Model] = None,
        critic_regularization_param: float = 0.0,
        epsilon_random_parameter: Optional[float] = None,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        """Instantiate RQLScenario.

        Args:
            running_objective (RunningObjective): The objective function
                to assess the costs over the prediction horizon.
            simulator (Simulator): The environment simulation for
                applying and testing the agent.
            prediction_horizon (int): The number of steps into the
                future over which predictions are made.
            critic_td_n (int): The n-step temporal-difference parameter
                used for critic updates.
            critic_batch_size (int): The batch size for the critic.
            predictor (Optional[Predictor]): The prediction model used
                for forecasting future states. Defaults to None.
            sampling_time (float): The time step interval for scenario.
                Defaults to 0.1.
            observer (Optional[Observer]): The observer object used for
                the algorithm. Defaults to None.
            constraint_parser (Optional[ConstraintParser]): The
                component for estimating the system's current state.
                Defaults to None.
            N_iterations (int): The number of iterations for the
                algorithm. Defaults to 1.
            discount_factor (float): The factor for discounting the
                value of future costs. Defaults to 1.0.
            policy_model (Optional[Model]): The model parameterizing the
                policy. Defaults to None. If `None` then
                `ModelWeightContainer` is used.
            critic_model (Optional[Model]): The model parameterizing the
                critic. Defaults to None. If `None` then diagonal
                quadratic form is used.
            critic_regularization_param (float): The regularization
                parameter for the critic. Defaults to 0.

        Returns:
            None: None
        """
        super().__init__(
            **get_predictive_kwargs(
                stopping_criterion=stopping_criterion,
                simulator=simulator,
                running_objective=running_objective,
                prediction_horizon=prediction_horizon,
                predictor=predictor,
                sampling_time=sampling_time,
                observer=observer,
                constraint_parser=constraint_parser,
                N_iterations=N_iterations,
                discount_factor=discount_factor,
                algorithm_name="rql",
                policy_model=policy_model,
                critic_model=critic_model,
                critic_regularization_param=critic_regularization_param,
                critic_td_n=critic_td_n,
                critic_is_on_policy=True,
                critic_is_value_function=False,
                critic_is_same_critic=False,
                critic_n_samples=critic_batch_size,
                epsilon_random_parameter=epsilon_random_parameter,
            )
        )


class RPV(RLScenario):
    """Implements Reward + Value algorithm."""

    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        critic_td_n: int,
        critic_batch_size: int,
        predictor: Optional[Predictor] = None,
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        N_iterations: int = 1,
        discount_factor: float = 1.0,
        policy_model: Optional[Model] = None,
        critic_model: Optional[Model] = None,
        critic_regularization_param: float = 0.0,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        """Instantiate RPV.

        Args:
            running_objective (RunningObjective): The objective function
                to assess the costs over the prediction horizon.
            simulator (Simulator): The environment simulation for
                applying and testing the agent.
            prediction_horizon (int): The number of steps into the
                future over which predictions are made.
            critic_td_n (int): The n-step temporal-difference parameter
                used for critic updates.
            critic_batch_size (int): The batch size for the critic.
            predictor (Optional[Predictor]): The prediction model used
                for forecasting future states. Defaults to None.
            sampling_time (float): The time step interval for scenario.
                Defaults to 0.1.
            observer (Optional[Observer]): The observer object used for
                the algorithm. Defaults to None.
            constraint_parser (Optional[ConstraintParser]): The
                component for estimating the system's current state.
                Defaults to None.
            N_iterations (int): The number of iterations for the
                algorithm. Defaults to 1.
            discount_factor (float): The factor for discounting the
                value of future costs. Defaults to 1.0.
            policy_model (Optional[Model]): The model parameterizing the
                policy. Defaults to None. If `None` then
                `ModelWeightContainer` is used.
            critic_model (Optional[Model]): The model parameterizing the
                critic. Defaults to None. If `None` then diagonal
                quadratic form is used.
            critic_regularization_param (float): The regularization
                parameter for the critic. Defaults to 0.

        Returns:
            None: None
        """
        super().__init__(
            **get_predictive_kwargs(
                stopping_criterion=stopping_criterion,
                simulator=simulator,
                running_objective=running_objective,
                prediction_horizon=prediction_horizon,
                predictor=predictor,
                sampling_time=sampling_time,
                observer=observer,
                constraint_parser=constraint_parser,
                N_iterations=N_iterations,
                discount_factor=discount_factor,
                algorithm_name="rpv",
                policy_model=policy_model,
                critic_model=critic_model,
                critic_regularization_param=critic_regularization_param,
                critic_td_n=critic_td_n,
                critic_is_on_policy=True,
                critic_is_value_function=True,
                critic_is_same_critic=False,
                critic_n_samples=critic_batch_size,
            )
        )


class MPCTorch(RLScenario):
    """MPCTorchScenario encapsulates the model predictive control (MPC) approach using PyTorch optimization for reinforcement learning.

    It integrates a policy that employs model-based predictions over a specified horizon to optimize actions, taking into account the dynamic nature of the environment. The scenario coordinates the interaction between the policy, model, critic, and environment.
    """

    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        sampling_time: float,
        n_epochs: int,
        opt_method_kwargs: Dict[str, Any],
        opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        predictor: Optional[Predictor] = None,
        model: Optional[ModelNN] = None,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        discount_factor: float = 1.0,
    ):
        """Initialize the object with the given parameters.

        Args:
            running_objective (RunningObjective): The objective function
                to assess the costs over the prediction horizon.
            simulator (Simulator): The environment simulation for
                applying and testing the agent.
            prediction_horizon (int): The number of steps into the
                future over which predictions are made.
            sampling_time (float): The time step interval for scenario.
            n_epochs (int): The number of training epochs.
            opt_method_kwargs (Dict[str, Any]): Additional keyword
                arguments for the optimization method.
            opt_method (Type[torch.optim.Optimizer]): The optimization
                method to use. Defaults to torch.optim.Adam.
            predictor (Optional[Predictor]): The predictor object to
                use. Defaults to None.
            model (Optional[ModelNN]): The neural network model
                parameterizing the policy. Defaults to None. If `None`
                then `ModelWeightContainerTorch` is used, but you can
                implement any neural network you want.
            observer (Optional[Observer]): Object responsible for state
                estimation from observations. Defaults to None.
            constraint_parser (Optional[ConstraintParser]): The
                mechanism for enforcing operational constraints.
                Defaults to None.
            discount_factor (float): The discount factor for future
                costs. Defaults to 1.0.

        Returns:
            None
        """
        system = simulator.system
        super().__init__(
            simulator=simulator,
            N_episodes=1,
            N_iterations=1,
            policy_optimization_event=Event.compute_action,
            critic=CriticTrivial(),
            running_objective=running_objective,
            observer=observer,
            sampling_time=sampling_time,
            policy=RLPolicy(
                action_bounds=system.action_bounds,
                model=(
                    ModelWeightContainerTorch(
                        dim_weights=(prediction_horizon + 1, system.dim_inputs),
                        output_bounds=system.action_bounds,
                    )
                    if model is None
                    else model
                ),
                constraint_parser=constraint_parser,
                system=system,
                running_objective=running_objective,
                prediction_horizon=prediction_horizon,
                algorithm="mpc",
                critic=CriticTrivial(),
                predictor=(
                    predictor
                    if predictor is not None
                    else EulerPredictor(system=system, pred_step_size=sampling_time)
                ),
                discount_factor=discount_factor,
                optimizer_config=TorchOptimizerConfig(
                    n_epochs=n_epochs,
                    opt_method=opt_method,
                    opt_method_kwargs=opt_method_kwargs,
                    data_buffer_iter_bathes_kwargs=dict(
                        batch_sampler=RollingBatchSampler,
                        dtype=torch.FloatTensor,
                        mode="backward",
                        batch_size=1,
                    ),
                ),
            ),
        )


def get_predictive_torch_kwargs(
    running_objective: RunningObjective,
    simulator: Simulator,
    prediction_horizon: int,
    predictor: Predictor,
    sampling_time: float,
    observer: Optional[Observer],
    constraint_parser: Optional[ConstraintParser],
    N_iterations: int,
    discount_factor: float,
    algorithm_name: str,
    policy_model: Optional[ModelNN] = None,
    policy_n_epochs: Optional[int] = None,
    policy_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
    policy_opt_method_kwargs: Optional[Dict[str, Any]] = None,
    critic_td_n: Optional[int] = None,
    critic_is_on_policy: Optional[bool] = None,
    critic_is_value_function: Optional[bool] = None,
    critic_is_same_critic: Optional[bool] = None,
    critic_model: Optional[ModelNN] = None,
    critic_n_epochs: Optional[int] = None,
    critic_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
    critic_opt_method_kwargs: Optional[Dict[str, Any]] = None,
    critic_batch_size: Optional[int] = None,
    epsilon_random_parameter: Optional[float] = None,
    size_mesh: Optional[int] = None,
    stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
):
    system = simulator.system
    critic = Critic(
        system=system,
        model=critic_model,
        td_n=critic_td_n,
        is_on_policy=critic_is_on_policy,
        is_value_function=critic_is_value_function,
        sampling_time=sampling_time,
        discount_factor=discount_factor,
        action_bounds=system.action_bounds,
        is_same_critic=critic_is_same_critic,
        optimizer_config=TorchOptimizerConfig(
            n_epochs=critic_n_epochs,
            opt_method=critic_opt_method,
            opt_method_kwargs=critic_opt_method_kwargs,
            data_buffer_iter_bathes_kwargs=dict(
                batch_sampler=RollingBatchSampler,
                dtype=torch.FloatTensor,
                mode="backward",
                batch_size=critic_batch_size,
            ),
        ),
        size_mesh=size_mesh,
    )

    if algorithm_name == "greedy":
        predictor_argument = None
    elif predictor is None:
        predictor_argument = EulerPredictor(system=system, pred_step_size=sampling_time)
    else:
        predictor_argument = predictor

    policy = RLPolicy(
        action_bounds=system.action_bounds,
        model=(
            ModelWeightContainerTorch(
                dim_weights=(
                    prediction_horizon + (1 if algorithm_name != "rpv" else 0),
                    system.dim_inputs,
                ),
                output_bounds=system.action_bounds,
                output_bounding_type="clip",
            )
            if policy_model is None
            else policy_model
        ),
        constraint_parser=constraint_parser,
        system=system,
        running_objective=running_objective,
        prediction_horizon=prediction_horizon,
        algorithm=algorithm_name,
        critic=critic,
        predictor=predictor_argument,
        discount_factor=discount_factor,
        optimizer_config=TorchOptimizerConfig(
            n_epochs=policy_n_epochs,
            opt_method=policy_opt_method,
            opt_method_kwargs=policy_opt_method_kwargs,
            data_buffer_iter_bathes_kwargs=dict(
                batch_sampler=RollingBatchSampler,
                dtype=torch.FloatTensor,
                mode="backward",
                batch_size=1,
                n_batches=1,
            ),
        ),
        epsilon_random_parameter=epsilon_random_parameter,
    )

    return dict(
        N_episodes=1,
        N_iterations=N_iterations,
        simulator=simulator,
        policy_optimization_event=Event.compute_action,
        critic_optimization_event=Event.compute_action,
        critic=critic,
        running_objective=running_objective,
        observer=observer,
        sampling_time=sampling_time,
        policy=policy,
        is_critic_first=True,
        stopping_criterion=stopping_criterion,
    )


class SQLTorch(RLScenario):
    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        critic_td_n: int,
        critic_batch_size: int,
        policy_n_epochs: int,
        critic_n_epochs: int,
        policy_opt_method_kwargs: Optional[Dict[str, Any]],
        critic_opt_method_kwargs: Optional[Dict[str, Any]],
        predictor: Optional[Predictor] = None,
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        N_iterations: int = 1,
        discount_factor: float = 1.0,
        policy_model: Optional[ModelNN] = None,
        policy_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_model: Optional[ModelNN] = None,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        super().__init__(
            **get_predictive_torch_kwargs(
                running_objective=running_objective,
                simulator=simulator,
                prediction_horizon=prediction_horizon,
                predictor=predictor,
                sampling_time=sampling_time,
                observer=observer,
                constraint_parser=constraint_parser,
                N_iterations=N_iterations,
                discount_factor=discount_factor,
                algorithm_name="sql",
                policy_model=policy_model,
                policy_n_epochs=policy_n_epochs,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_is_on_policy=True,
                critic_is_value_function=False,
                critic_is_same_critic=False,
                critic_model=critic_model,
                critic_n_epochs=critic_n_epochs,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_batch_size=critic_batch_size,
                stopping_criterion=stopping_criterion,
            )
        )


class RQLTorch(RLScenario):
    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        critic_td_n: int,
        critic_batch_size: int,
        policy_n_epochs: int,
        critic_n_epochs: int,
        policy_opt_method_kwargs: Optional[Dict[str, Any]],
        critic_opt_method_kwargs: Optional[Dict[str, Any]],
        predictor: Optional[Predictor] = None,
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        N_iterations: int = 1,
        discount_factor: float = 1.0,
        policy_model: Optional[ModelNN] = None,
        policy_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_model: Optional[ModelNN] = None,
        epsilon_random_parameter: Optional[float] = None,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        super().__init__(
            **get_predictive_torch_kwargs(
                running_objective=running_objective,
                simulator=simulator,
                prediction_horizon=prediction_horizon,
                predictor=predictor,
                sampling_time=sampling_time,
                observer=observer,
                constraint_parser=constraint_parser,
                N_iterations=N_iterations,
                discount_factor=discount_factor,
                algorithm_name="rql",
                policy_model=policy_model,
                policy_n_epochs=policy_n_epochs,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_is_on_policy=True,
                critic_is_value_function=False,
                critic_is_same_critic=False,
                critic_model=critic_model,
                critic_n_epochs=critic_n_epochs,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_batch_size=critic_batch_size,
                epsilon_random_parameter=epsilon_random_parameter,
                stopping_criterion=stopping_criterion,
            )
        )


class RPVTorch(RLScenario):
    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        critic_td_n: int,
        critic_batch_size: int,
        policy_n_epochs: int,
        critic_n_epochs: int,
        policy_opt_method_kwargs: Optional[Dict[str, Any]],
        critic_opt_method_kwargs: Optional[Dict[str, Any]],
        predictor: Optional[Predictor] = None,
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        N_iterations: int = 1,
        discount_factor: float = 1.0,
        policy_model: Optional[ModelNN] = None,
        policy_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_model: Optional[ModelNN] = None,
        epsilon_random_parameter: Optional[float] = None,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        super().__init__(
            **get_predictive_torch_kwargs(
                running_objective=running_objective,
                simulator=simulator,
                prediction_horizon=prediction_horizon,
                predictor=predictor,
                sampling_time=sampling_time,
                observer=observer,
                constraint_parser=constraint_parser,
                N_iterations=N_iterations,
                discount_factor=discount_factor,
                algorithm_name="rpv",
                policy_model=policy_model,
                policy_n_epochs=policy_n_epochs,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_is_on_policy=True,
                critic_is_value_function=True,
                critic_is_same_critic=False,
                critic_model=critic_model,
                critic_n_epochs=critic_n_epochs,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_batch_size=critic_batch_size,
                epsilon_random_parameter=epsilon_random_parameter,
                stopping_criterion=stopping_criterion,
            )
        )


class SARSA(RLScenario):
    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        critic_td_n: int,
        critic_batch_size: int,
        policy_n_epochs: int,
        critic_n_epochs: int,
        critic_model: Optional[ModelNN],
        policy_opt_method_kwargs: Optional[Dict[str, Any]],
        critic_opt_method_kwargs: Optional[Dict[str, Any]],
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        N_iterations: int = 1,
        discount_factor: float = 1.0,
        policy_model: Optional[ModelNN] = None,
        policy_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        epsilon_random_parameter: Optional[float] = None,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        super().__init__(
            **get_predictive_torch_kwargs(
                running_objective=running_objective,
                simulator=simulator,
                prediction_horizon=0,
                predictor=None,
                sampling_time=sampling_time,
                observer=observer,
                constraint_parser=constraint_parser,
                N_iterations=N_iterations,
                discount_factor=discount_factor,
                algorithm_name="greedy",
                policy_model=policy_model,
                policy_n_epochs=policy_n_epochs,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_is_on_policy=True,
                critic_is_value_function=False,
                critic_is_same_critic=False,
                critic_model=critic_model,
                critic_n_epochs=critic_n_epochs,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_batch_size=critic_batch_size,
                epsilon_random_parameter=epsilon_random_parameter,
                stopping_criterion=stopping_criterion,
            )
        )


class DQN(RLScenario):
    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        critic_td_n: int,
        critic_batch_size: int,
        policy_n_epochs: int,
        critic_n_epochs: int,
        critic_model: Optional[ModelNN],
        policy_opt_method_kwargs: Optional[Dict[str, Any]],
        critic_opt_method_kwargs: Optional[Dict[str, Any]],
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        N_iterations: int = 1,
        discount_factor: float = 1.0,
        policy_model: Optional[ModelNN] = None,
        policy_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        critic_opt_method: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        epsilon_random_parameter: Optional[float] = None,
        size_mesh: int = 100,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        super().__init__(
            **get_predictive_torch_kwargs(
                running_objective=running_objective,
                simulator=simulator,
                prediction_horizon=0,
                predictor=None,
                sampling_time=sampling_time,
                observer=observer,
                constraint_parser=constraint_parser,
                N_iterations=N_iterations,
                discount_factor=discount_factor,
                algorithm_name="greedy",
                policy_model=policy_model,
                policy_n_epochs=policy_n_epochs,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_is_on_policy=False,
                critic_is_value_function=False,
                critic_is_same_critic=False,
                critic_model=critic_model,
                critic_n_epochs=critic_n_epochs,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_batch_size=critic_batch_size,
                epsilon_random_parameter=epsilon_random_parameter,
                size_mesh=size_mesh,
                stopping_criterion=stopping_criterion,
            )
        )


def get_policy_gradient_kwargs(
    sampling_time: float,
    running_objective: RunningObjective,
    simulator: Simulator,
    discount_factor: float,
    observer: Optional[Observer],
    N_episodes: int,
    N_iterations: int,
    value_threshold: float,
    policy_type: Type[Policy],
    policy_model: PerceptronWithTruncatedNormalNoise,
    policy_n_epochs: int,
    policy_opt_method_kwargs: Dict[str, Any],
    policy_opt_method: Type[torch.optim.Optimizer],
    is_reinstantiate_policy_optimizer: bool,
    critic_model: Optional[ModelPerceptron] = None,
    critic_opt_method: Optional[Type[torch.optim.Optimizer]] = None,
    critic_opt_method_kwargs: Optional[Dict[str, Any]] = None,
    critic_n_epochs: Optional[int] = None,
    critic_td_n: Optional[int] = None,
    critic_kwargs: Dict[str, Any] = None,
    critic_is_value_function: Optional[bool] = None,
    is_reinstantiate_critic_optimizer: Optional[bool] = None,
    policy_kwargs: Dict[str, Any] = None,
    scenario_kwargs: Dict[str, Any] = None,
    is_use_critic_as_policy_kwarg: bool = True,
    stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    device: str = "cpu",
):
    system = simulator.system
    if critic_model is not None:
        assert (
            critic_n_epochs is not None
            and critic_td_n is not None
            and critic_opt_method is not None
            and critic_opt_method_kwargs is not None
            and is_reinstantiate_critic_optimizer is not None
            and critic_is_value_function is not None
        ), "critic_n_epochs, critic_td_n, critic_opt_method, critic_opt_method_kwargs, is_reinstantiate_critic_optimizer, critic_is_value_function should be set"
        critic = Critic(
            system=system,
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            model=critic_model,
            td_n=critic_td_n,
            device=device,
            is_value_function=critic_is_value_function,
            is_on_policy=True,
            is_same_critic=True,
            optimizer_config=TorchOptimizerConfig(
                critic_n_epochs,
                data_buffer_iter_bathes_kwargs={
                    "batch_sampler": RollingBatchSampler,
                    "dtype": torch.FloatTensor,
                    "mode": "full",
                    "n_batches": 1,
                    "device": device,
                },
                opt_method_kwargs=critic_opt_method_kwargs,
                opt_method=critic_opt_method,
                is_reinstantiate_optimizer=is_reinstantiate_critic_optimizer,
            ),
            is_full_iteration_epoch=True,
            **(critic_kwargs if critic_kwargs is not None else dict()),
        )
    else:
        critic = CriticTrivial()

    return dict(
        stopping_criterion=stopping_criterion,
        simulator=simulator,
        discount_factor=discount_factor,
        policy_optimization_event=Event.reset_iteration,
        critic_optimization_event=Event.reset_iteration,
        N_episodes=N_episodes,
        N_iterations=N_iterations,
        running_objective=running_objective,
        observer=observer,
        critic=critic,
        is_critic_first=True,
        value_threshold=value_threshold,
        sampling_time=sampling_time,
        policy=policy_type(
            model=policy_model,
            system=system,
            device=device,
            discount_factor=discount_factor,
            optimizer_config=TorchOptimizerConfig(
                n_epochs=policy_n_epochs,
                opt_method=policy_opt_method,
                opt_method_kwargs=policy_opt_method_kwargs,
                is_reinstantiate_optimizer=is_reinstantiate_policy_optimizer,
                data_buffer_iter_bathes_kwargs={
                    "batch_sampler": RollingBatchSampler,
                    "dtype": torch.FloatTensor,
                    "mode": "full",
                    "n_batches": 1,
                    "device": device,
                },
            ),
            **(dict(critic=critic) if is_use_critic_as_policy_kwarg else dict()),
            **(policy_kwargs if policy_kwargs is not None else dict()),
        ),
        **(scenario_kwargs if scenario_kwargs is not None else dict()),
    )


class PPO(RLScenario):
    """Scenario for Proximal Polizy Optimization.

    PPOScenario is a reinforcement learning scenario implementing the Proximal Policy Optimization (PPO) algorithm.
    This algorithm uses a policy gradient approach with an objective function designed to reduce the variance
    of policy updates, while ensuring that the new policy does not deviate significantly from the old one.
    """

    def __init__(
        self,
        policy_model: PerceptronWithTruncatedNormalNoise,
        critic_model: ModelPerceptron,
        sampling_time: float,
        running_objective: RunningObjective,
        simulator: Simulator,
        critic_n_epochs: int,
        policy_n_epochs: int,
        critic_opt_method_kwargs: Dict[str, Any],
        policy_opt_method_kwargs: Dict[str, Any],
        critic_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        running_objective_type: str = "cost",
        critic_td_n: int = 1,
        cliprange: float = 0.2,
        discount_factor: float = 0.7,
        observer: Optional[Observer] = None,
        N_episodes: int = 2,
        N_iterations: int = 100,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
        gae_lambda: float = 0.0,
        is_normalize_advantages: bool = True,
        device: str = "cpu",
        entropy_coeff: float = 0.0,
    ):
        """Initialize the object with the given parameters.

        Args:
            policy_model (PerceptronWithTruncatedNormalNoise): The
                neural network model parameterizing the policy.
            critic_model (ModelPerceptron): The neural network model
                used for the value function approximation.
            sampling_time (float): Time interval between two consecutive
                actions.
            running_objective (RunningObjective): A function that
                returns the scalar running cost or reward associated
                with an observation-action pair.
            simulator (Simulator): The simulation environment where the
                agent performs actions.
            critic_n_epochs (int): The number of epochs for which the
                critic is trained per iteration.
            policy_n_epochs (int): The number of epochs for which the
                policy is trained per iteration.
            critic_opt_method_kwargs (Dict[str, Any]): A dictionary of
                keyword arguments for the optimizer used for the critic.
            policy_opt_method_kwargs (Dict[str, Any]): A dictionary of
                keyword arguments for the optimizer used for the policy.
            critic_opt_method (Type[torch.optim.Optimizer]): The
                optimization algorithm class used for training the
                critic, e.g. torch.optim.Adam.
            policy_opt_method (Type[torch.optim.Optimizer]): The
                optimization algorithm class used for training the
                policy, e.g. torch.optim.Adam.
            running_objective_type (str): Specifies whether the running
                objective represents a 'cost' to minimize or a 'reward'
                to maximize.
            critic_td_n (int): The n-step temporal-difference parameter
                used for critic updates.
            epsilon (float): Clipping parameter that restricts the
                deviation of the new policy from the old policy.
            discount_factor (float): A factor applied to future rewards
                or costs to discount their value relative to immediate
                ones.
            observer (Optional[Observer]): Object responsible for state
                estimation from observations.
            N_episodes (int): The number of episodes to run in every
                iteration.
            N_iterations (int): The number of iterations to run in the
                scenario.
            value_threshold (float): Threshold of the value to
                end an episode.

        Raises:
            AssertionError: If the `running_objective_type` is invalid.

        Returns:
            None
        """
        assert (
            running_objective_type == "cost" or running_objective_type == "reward"
        ), f"Invalid 'running_objective_type' value: '{running_objective_type}'. It must be either 'cost' or 'reward'."
        super().__init__(
            **get_policy_gradient_kwargs(
                sampling_time=sampling_time,
                running_objective=running_objective,
                simulator=simulator,
                discount_factor=discount_factor,
                observer=observer,
                N_episodes=N_episodes,
                N_iterations=N_iterations,
                value_threshold=value_threshold,
                policy_type=PolicyPPO,
                policy_model=policy_model,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                is_reinstantiate_policy_optimizer=True,
                policy_kwargs=dict(
                    cliprange=cliprange,
                    running_objective_type=running_objective_type,
                    sampling_time=sampling_time,
                    gae_lambda=gae_lambda,
                    is_normalize_advantages=is_normalize_advantages,
                    entropy_coeff=entropy_coeff,
                ),
                policy_n_epochs=policy_n_epochs,
                critic_model=critic_model,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_n_epochs=critic_n_epochs,
                critic_is_value_function=True,
                is_reinstantiate_critic_optimizer=True,
                stopping_criterion=stopping_criterion,
                device=device,
            )
        )


class SDPG(RLScenario):
    """Implements the Stochastic Deep Policy Gradient (SDPG) algorithm.

    SDPGScenario implements the Stochastic Deep Policy Gradient (SDPG) algorithm,
    an off-policy actor-critic method using a deep policy to model stochastic policies and
    a deep network as the critic that approximates a value Function.

    The SDPG algorithm performs policy updates by maximizing the critic's output and updates
    the critic using temporal-difference learning. This scenario orchestrates the training
    process, including interaction with the simulation environment, handling of the experience
    replay buffer, and coordination of policy and critic improvements.
    """

    def __init__(
        self,
        policy_model: PerceptronWithTruncatedNormalNoise,
        critic_model: ModelPerceptron,
        sampling_time: float,
        running_objective: RunningObjective,
        simulator: Simulator,
        critic_n_epochs: int,
        policy_n_epochs: int,
        critic_opt_method_kwargs: Dict[str, Any],
        policy_opt_method_kwargs: Dict[str, Any],
        critic_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        critic_td_n: int = 1,
        discount_factor: float = 0.7,
        observer: Optional[Observer] = None,
        N_episodes: int = 2,
        N_iterations: int = 100,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
        is_normalize_advantages=True,
        gae_lambda=0.0,
    ):
        """Initialize SDPG.

        Args:
            policy_model: The
                policy network model that defines the policy
                architecture.
            critic_model: The critic network model
                that defines the value function architecture.
            sampling_time (float): The time step between agent actions
                in the environment.
            running_objective (RunningObjective): Function calculating
                the reward or cost at each time step when an action is
                taken.
            simulator (Simulator): The environment in which the agent
                operates, providing observation.
            critic_n_epochs (int): Number of epochs for which the critic
                network is trained at each iteration.
            policy_n_epochs (int): Number of epochs for which the policy
                network is trained at each iteration.
            critic_opt_method_kwargs (Dict[str, Any]): Parameters for
                the critic's optimization algorithm.
            policy_opt_method_kwargs (Dict[str, Any]): Parameters for
                the policy's optimization algorithm.
            critic_opt_method (Type[torch.optim.Optimizer]): Class of
                the optimizer to use for optimizing the critic,
                defaults to torch.optim.Adam
            policy_opt_method (Type[torch.optim.Optimizer]): Class of
                the optimizer to use for optimizing the policy, ,
                defaults to torch.optim.Adam
            critic_td_n (int): The number of steps to look ahead in the
                TD-target for the critic.
            discount_factor (float): The factor by which future rewards
                or costs are discounted relative to immediate running
                objectives.
            observer (Optional[Observer]): The observer object that
                estimates the state of the environment from
                observations.
            N_episodes (int): The number of episodes per iteration.
            N_iterations (int): The total number of iterations for
                training.
            value_threshold (float): Threshold for the value
                that, once reached, stops the episode, defaults to
                np.inf.
        """
        super().__init__(
            **get_policy_gradient_kwargs(
                sampling_time=sampling_time,
                running_objective=running_objective,
                simulator=simulator,
                discount_factor=discount_factor,
                observer=observer,
                N_episodes=N_episodes,
                N_iterations=N_iterations,
                value_threshold=value_threshold,
                policy_type=PolicySDPG,
                policy_model=policy_model,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                is_reinstantiate_policy_optimizer=False,
                policy_n_epochs=policy_n_epochs,
                critic_model=critic_model,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_n_epochs=critic_n_epochs,
                critic_is_value_function=True,
                is_reinstantiate_critic_optimizer=True,
                policy_kwargs=dict(
                    sampling_time=sampling_time,
                    is_normalize_advantages=is_normalize_advantages,
                    gae_lambda=gae_lambda,
                ),
                stopping_criterion=stopping_criterion,
            )
        )


class REINFORCE(RLScenario):
    """Implements the REINFORCE algorithm."""

    def __init__(
        self,
        policy_model: PerceptronWithTruncatedNormalNoise,
        sampling_time: float,
        running_objective: RunningObjective,
        simulator: Simulator,
        policy_opt_method_kwargs: Dict[str, Any],
        policy_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_n_epochs: int = 1,
        discount_factor: float = 1.0,
        observer: Optional[Observer] = None,
        N_episodes: int = 4,
        N_iterations: int = 100,
        value_threshold: float = np.inf,
        is_with_baseline: bool = True,
        is_do_not_let_the_past_distract_you: bool = True,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        """Initialize an REINFORCE object.

        Args:
            policy_model: The
                policy network model that defines the policy
                architecture.
            sampling_time: The time step between agent actions
                in the environment.
            running_objective: Function calculating
                the reward or cost at each time step when an action is
                taken.
            simulator: The environment in which the agent
                operates, providing state, observation.
            policy_opt_method_kwargs: The keyword
                arguments for the policy optimizer method.
            policy_opt_method (Type[torch.optim.Optimizer], optional):
                The policy optimizer method.
            n_epochs: The number of epochs used by the
                policy optimizer.
            discount_factor: The discount factor used
                by the RLScenario. Defaults to 1.0.
            observer: The observer object
                that estimates the state of the environment from observations.
            N_episodes: The number of episodes per iteration.
            N_iterations: The total number of iterations
                for training.
            is_with_baseline: Whether to use baseline as value (i.e. cumulative cost or reward)
                from previous iteration.
            is_do_not_let_the_past_distract_you: Whether to use tail total costs or not.

        Returns:
            None: None
        """
        super().__init__(
            **get_policy_gradient_kwargs(
                sampling_time=sampling_time,
                running_objective=running_objective,
                simulator=simulator,
                discount_factor=discount_factor,
                observer=observer,
                N_episodes=N_episodes,
                N_iterations=N_iterations,
                value_threshold=value_threshold,
                policy_type=PolicyReinforce,
                policy_model=policy_model,
                policy_n_epochs=policy_n_epochs,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                is_reinstantiate_policy_optimizer=False,
                policy_kwargs=dict(
                    is_with_baseline=is_with_baseline,
                    is_do_not_let_the_past_distract_you=is_do_not_let_the_past_distract_you,
                ),
                is_use_critic_as_policy_kwarg=False,
                stopping_criterion=stopping_criterion,
            ),
        )


class DDPG(RLScenario):
    """Implements a scenario for interacting with an environment using the Deep Deterministic Policy Gradients (DDPG) algorithm."""

    def __init__(
        self,
        policy_model: PerceptronWithTruncatedNormalNoise,
        critic_model: ModelPerceptron,
        sampling_time: float,
        running_objective: RunningObjective,
        simulator: Simulator,
        critic_n_epochs: int,
        policy_n_epochs: int,
        critic_opt_method_kwargs: Dict[str, Any],
        policy_opt_method_kwargs: Dict[str, Any],
        critic_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        critic_td_n: int = 1,
        discount_factor: float = 0.7,
        observer: Optional[Observer] = None,
        N_episodes: int = 2,
        N_iterations: int = 100,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        """Instantiate DDPG Scenario.

        Args:
            policy_model (PerceptronWithTruncatedNormalNoise): The
                policy (actor) neural network model with input as state
                and output as action.
            critic_model (ModelPerceptron): The critic neural network
                model with input as state-action pair and output as
                Q-value estimate.
            sampling_time (float): The time interval between each action
                taken by the policy.
            running_objective (RunningObjective): Th function
                calculating the running cost at each time step when an
                action is taken.
            simulator (Simulator): The environment simulator in which
                the agent operates.
            critic_n_epochs (int): The number of epochs for training the
                critic model during each optimization.
            policy_n_epochs (int): The number of epochs for training the
                policy model during each optimization.
            critic_opt_method_kwargs (Dict[str, Any]): Keyword arguments
                for the critic optimizer method.
            policy_opt_method_kwargs (Dict[str, Any]): Keyword arguments
                for the policy optimizer method.
            critic_opt_method (Type[torch.optim.Optimizer]): The
                optimizer class to be used for the critic. Defaults to
                torch.nn.Adam.
            policy_opt_method (Type[torch.optim.Optimizer]): The
                optimizer class to be used for the policy. Defaults to
                torch.nn.Adam.
            critic_td_n (int): The n-step return for temporal-difference
                learning for the critic estimator.
            discount_factor (float): The discount factor that weighs
                future costs lower compared to immediate costs.
            observer (Optional[Observer]): An observer object used for
                deriving state estimations from raw observations.
            N_episodes (int): The number of episodes to be executed in
                each training iteration.
            N_iterations (int): The total number of training iterations
                for the scenario.
            value_threshold (float): The threshold for the total
                cumulative objective value that triggers the end of an
                episode.
        """
        super().__init__(
            **get_policy_gradient_kwargs(
                sampling_time=sampling_time,
                running_objective=running_objective,
                simulator=simulator,
                discount_factor=discount_factor,
                observer=observer,
                N_episodes=N_episodes,
                N_iterations=N_iterations,
                value_threshold=value_threshold,
                policy_type=PolicyDDPG,
                policy_model=policy_model,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                is_reinstantiate_policy_optimizer=True,
                policy_n_epochs=policy_n_epochs,
                critic_model=critic_model,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_n_epochs=critic_n_epochs,
                critic_is_value_function=False,
                is_reinstantiate_critic_optimizer=True,
                stopping_criterion=stopping_criterion,
            )
        )


# Additional classes such as Policy may need to be defined or imported depending on the setup of your framework.
