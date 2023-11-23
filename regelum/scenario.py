"""Contains high-level structures of scenarios (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import torch

from .__utilities import Clock, AwaitedParameter
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

        :param policy: Policy to generate actions based on observations.
        :param simulator: Simulator to interact with and collect data for training.
        :param sampling_time: Time interval between action updates.
        :param running_objective: Objective function for evaluating performance.
        :param constraint_parser: Tool for parsing constraints during policy optimization.
        :param observer: Observer for estimating the system state.
        :param N_episodes: Total number of episodes to run.
        :param N_iterations: Total number of iterations to run.
        :param value_threshold: Threshold to stop the simulation if the objective is met.
        :param discount_factor: Discount factor for future rewards.
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

                self.reload_scenario()
            if self.sim_status == "simulation_ended":
                break

    def step(self):
        if isinstance(self.action_init, AwaitedParameter) and isinstance(
            self.state_init, AwaitedParameter
        ):
            (
                self.state_init,
                self.action_init,
            ) = self.simulator.get_init_state_and_action()

        if not self.is_episode_ended and self.value <= self.value_threshold:
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

                is_simulation_ended = (
                    self.iteration_counter >= self.N_iterations
                    or self.sim_status == "simulation_ended"
                )

                if is_simulation_ended:
                    return "simulation_ended"
                else:
                    return "iteration_ended"
            else:
                return "episode_ended"

    @apply_callbacks()
    def reset_iteration(self):
        pass

    def reset_episode(self):
        self.is_episode_ended = False
        return self.value

    @apply_callbacks()
    def reload_scenario(self):
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
            "timestamp": self.time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
            "action": self.policy.action,
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
        return self.policy.action

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
            observation, self.policy.action
        )
        self.value = self.calculate_value(self.current_running_objective, self.time)
        observation_action = np.concatenate((observation, self.policy.action), axis=1)
        return {
            "action": self.policy.action,
            "running_objective": self.current_running_objective,
            "current_value": self.value,
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
        max_data_buffer_size: Optional[int] = None,
        sampling_time: float = 0.1,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        """Instantiate a RLScenario object.

        :param policy: Policy object
        :type policy: Policy
        :param critic: Cricic
        :type critic: Critic
        :param running_objective: Function to calculate the running objective.
        :type running_objective: RunningObjective
        :param critic_optimization_event: moments when to optimize critic. Can be either 'compute_action' for online learning, or 'reset_episode' for optimizing after each episode, or 'reset_iteration' for optimizing after each iteration
        :type critic_optimization_event: str
        :param policy_optimization_event: moments when to optimize critic. Can be either 'compute_action' for online learning, or 'reset_episode' for optimizing after each episode, or 'reset_iteration' for optimizing after each iteration
        :type policy_optimization_event: str
        :param discount_factor: Discount factor. Used for computing total objective as discounted sum (or integral) of running objectives, defaults to 1.0
        :type discount_factor: float, optional
        :param is_critic_first: if is True then critic is optimized first then policy (can be usefull in DQN or Predictive Algorithms such as RPV, RQL, SQL). For `False` firstly is policy optimized then critic. defaults to False
        :type is_critic_first: bool, optional
        :param action_bounds: action bounds. Applied for every generated action as clip, defaults to None
        :type action_bounds: Union[list, np.ndarray, None], optional
        :param max_data_buffer_size: max size of DataBuffer, if is `None` the DataBuffer is unlimited. defaults to None
        :type max_data_buffer_size: Optional[int], optional
        :param sampling_time: time interval between two consecutive actions, defaults to 0.1
        :type sampling_time: float, optional
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

        self.critic_optimization_event = critic_optimization_event
        self.policy_optimization_event = policy_optimization_event
        self.data_buffer = DataBuffer(max_data_buffer_size)
        self.critic = critic
        self.is_first_compute_action_call = True
        self.is_critic_first = is_critic_first
        self.stopping_criterion = stopping_criterion

    def _reset_whatever(self, suffix):
        def wrapped(*args, **kwargs):
            res = object.__getattribute__(self, f"reset_{suffix}")(*args, **kwargs)
            if self.sim_status != "simulation_ended":
                self.optimize(event=getattr(Event, f"reset_{suffix}"))
            if suffix == "iteration":
                self.data_buffer.nullify_buffer()
            return res

        return wrapped

    @apply_callbacks()
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
        self.safe_policy = safe_policy

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

    @apply_callbacks()
    def compute_action(
        self,
        estimated_state,
        observation,
        time=0,
    ):
        assert np.allclose(
            estimated_state, self.data_buffer.get_latest("estimated_state")
        )
        assert np.allclose(observation, self.data_buffer.get_latest("observation"))
        assert np.allclose(time, self.data_buffer.get_latest("timestamp"))

        if self.is_first_compute_action_call:
            self.critic.observation_last_good = observation
            self.issue_action(observation, is_safe=True)
            self.is_first_compute_action_call = False
        else:
            self.pre_optimize(self.critic, Event.compute_action, time)
            critic_weights = self.critic.optimize(
                self.data_buffer, is_update_and_cache_weights=False
            )
            critic_weights_accepted = self.critic.opt_status == OptStatus.success

            if critic_weights_accepted:
                self.critic.update_weights(critic_weights)
                self.pre_optimize(self.policy, Event.compute_action, time)
                self.policy.optimize(self.data_buffer)
                policy_weights_accepted = self.policy.opt_status == OptStatus.success
                if policy_weights_accepted:
                    self.critic.observation_last_good = observation
                    self.issue_action(observation, is_safe=False)
                    self.critic.cache_weights(critic_weights)
                else:
                    self.issue_action(observation, is_safe=True)
            else:
                self.issue_action(observation, is_safe=True)

        return self.policy.action


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
        N_iterations=5,
    ):
        """Instantiate CALF class.

        :param simulator: The simulator object.
        :type simulator: Simulator
        :param running_objective: The running objective.
        :type running_objective: RunningObjective
        :param safe_policy: The safe policy.
        :type safe_policy: Policy
        :param critic_td_n: The TD-N parameter for the critic. Defaults to 2.
        :type critic_td_n: int, optional
        :param observer: The observer object. Defaults to None.
        :type observer: Optional[Observer], optional
        :param prediction_horizon: The prediction horizon. Defaults to 1.
        :type prediction_horizon: int, optional
        :param policy_model: The policy model. Defaults to None.
        :type policy_model: Optional[Model], optional
        :param critic_model: The critic model. Defaults to None.
        :type critic_model: Optional[Model], optional
        :param predictor: The predictor object. Defaults to None.
        :type predictor: Optional[Predictor], optional
        :param discount_factor: The discount factor. Defaults to 1.0.
        :type discount_factor: float, optional
        :param sampling_time: The sampling time. Defaults to 0.1.
        :type sampling_time: float, optional
        :param critic_lb_parameter: The lower bound parameter for the critic. Defaults to 0.0.
        :type critic_lb_parameter: float, optional
        :param critic_ub_parameter: The upper bound parameter for the critic. Defaults to 1.0.
        :type critic_ub_parameter: float, optional
        :param critic_safe_decay_param: The safe decay parameter for the critic. Defaults to 0.001.
        :type critic_safe_decay_param: float, optional
        :param critic_is_dynamic_decay_rate: Whether the critic has a dynamic decay rate. Defaults to False.
        :type critic_is_dynamic_decay_rate: bool, optional
        :param critic_batch_size: The batch size for the critic optimizer. Defaults to 10.
        :type critic_batch_size: int, optional

        :returns: None
        :rtype: None
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
        )
        policy = RLPolicy(
            action_bounds=system.action_bounds,
            model=ModelWeightContainer(
                weights_init=np.zeros(
                    (prediction_horizon, system.dim_inputs),
                    dtype=np.float64,
                ),
                dim_output=system.dim_inputs,
            )
            if policy_model is None
            else policy_model,
            system=system,
            running_objective=running_objective,
            prediction_horizon=prediction_horizon,
            algorithm="rpv",
            critic=critic,
            predictor=predictor
            if predictor is not None
            else EulerPredictor(system=system, pred_step_size=sampling_time),
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
            model=ModelWeightContainerTorch(
                dim_weights=(prediction_horizon, system.dim_inputs),
                output_bounds=system.action_bounds,
                output_bounding_type="clip",
            )
            if policy_model is None
            else policy_model,
            system=system,
            running_objective=running_objective,
            prediction_horizon=prediction_horizon,
            algorithm="rpv",
            critic=critic,
            predictor=predictor
            if predictor is not None
            else EulerPredictor(system=system, pred_step_size=sampling_time),
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

        :param running_objective: The objective function to assess the costs over the prediction horizon.
        :type running_objective: RunningObjective
        :param simulator: The environment simulation for applying and testing the agent.
        :type simulator: Simulator
        :param prediction_horizon: The number of steps into the future over which predictions are made.
        :type prediction_horizon: int
        :param predictor: The prediction model used for forecasting future states.
        :type predictor: Optional[Predictor]
        :param sampling_time:  The time step interval for scenario.
        :type sampling_time: float
        :param observer: The component for estimating the system's current state. Defaults to None.
        :type observer: Observer | None
        :param constraint_parser: The mechanism for enforcing operational constraints. Defaults to None.
        :type constraint_parser: Optional[ConstraintParser]
        :param discount_factor: The factor for discounting the value of future costs. Defaults to 1.0.
        :type discount_factor: float
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
                predictor=predictor
                if predictor is not None
                else EulerPredictor(system=system, pred_step_size=sampling_time),
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
        model=critic_model
        if critic_model is not None
        else ModelQuadLin("diagonal", is_with_linear_terms=False),
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
        model=ModelWeightContainer(
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
        else policy_model,
        constraint_parser=constraint_parser,
        system=system,
        running_objective=running_objective,
        prediction_horizon=prediction_horizon,
        algorithm=algorithm_name,
        critic=critic,
        predictor=predictor
        if predictor is not None
        else EulerPredictor(system=system, pred_step_size=sampling_time),
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

        :param running_objective: The objective function to assess the costs over the prediction horizon.
        :type running_objective: RunningObjective
        :param simulator: The environment simulation for applying and testing the agent.
        :type simulator: Simulator
        :param prediction_horizon: The number of steps into the future over which predictions are made.
        :type prediction_horizon: int
        :param critic_td_n: The n-step temporal-difference parameter used for critic updates.
        :type critic_td_n: int
        :param critic_batch_size: The batch size for the critic.
        :type critic_batch_size: int
        :param predictor: The prediction model used for forecasting future states. Defaults to None.
        :type predictor: Optional[Predictor]
        :param sampling_time: The time step interval for scenario. Defaults to 0.1.
        :type sampling_time: float
        :param observer: The observer object used for the algorithm. Defaults to None.
        :type observer: Optional[Observer]
        :param constraint_parser: The component for estimating the system's current state. Defaults to None.
        :type constraint_parser: Optional[ConstraintParser]
        :param N_iterations: The number of iterations for the algorithm. Defaults to 1.
        :type N_iterations: int
        :param discount_factor: The factor for discounting the value of future costs. Defaults to 1.0.
        :type discount_factor: float
        :param policy_model: The model parameterizing the policy. Defaults to None. If `None` then `ModelWeightContainer` is used.
        :type policy_model: Optional[Model]
        :param critic_model:  The model parameterizing the critic. Defaults to None. If `None` then diagonal quadratic form is used.
        :type critic_model: Optional[Model]
        :param critic_regularization_param: The regularization parameter for the critic. Defaults to 0.
        :type critic_regularization_param: float

        :return: None
        :rtype: None
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

        :param running_objective: The objective function to assess the costs over the prediction horizon.
        :type running_objective: RunningObjective
        :param simulator: The environment simulation for applying and testing the agent.
        :type simulator: Simulator
        :param prediction_horizon: The number of steps into the future over which predictions are made.
        :type prediction_horizon: int
        :param critic_td_n: The n-step temporal-difference parameter used for critic updates.
        :type critic_td_n: int
        :param critic_batch_size: The batch size for the critic.
        :type critic_batch_size: int
        :param predictor: The prediction model used for forecasting future states. Defaults to None.
        :type predictor: Optional[Predictor]
        :param sampling_time: The time step interval for scenario. Defaults to 0.1.
        :type sampling_time: float
        :param observer: The observer object used for the algorithm. Defaults to None.
        :type observer: Optional[Observer]
        :param constraint_parser: The component for estimating the system's current state. Defaults to None.
        :type constraint_parser: Optional[ConstraintParser]
        :param N_iterations: The number of iterations for the algorithm. Defaults to 1.
        :type N_iterations: int
        :param discount_factor: The factor for discounting the value of future costs. Defaults to 1.0.
        :type discount_factor: float
        :param policy_model: The model parameterizing the policy. Defaults to None. If `None` then `ModelWeightContainer` is used.
        :type policy_model: Optional[Model]
        :param critic_model:  The model parameterizing the critic. Defaults to None. If `None` then diagonal quadratic form is used.
        :type critic_model: Optional[Model]
        :param critic_regularization_param: The regularization parameter for the critic. Defaults to 0.
        :type critic_regularization_param: float

        :return: None
        :rtype: None
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

        :param running_objective: The objective function to assess the costs over the prediction horizon.
        :type running_objective: RunningObjective
        :param simulator: The environment simulation for applying and testing the agent.
        :type simulator: Simulator
        :param prediction_horizon: The number of steps into the future over which predictions are made.
        :type prediction_horizon: int
        :param critic_td_n: The n-step temporal-difference parameter used for critic updates.
        :type critic_td_n: int
        :param critic_batch_size: The batch size for the critic.
        :type critic_batch_size: int
        :param predictor: The prediction model used for forecasting future states. Defaults to None.
        :type predictor: Optional[Predictor]
        :param sampling_time: The time step interval for scenario. Defaults to 0.1.
        :type sampling_time: float
        :param observer: The observer object used for the algorithm. Defaults to None.
        :type observer: Optional[Observer]
        :param constraint_parser: The component for estimating the system's current state. Defaults to None.
        :type constraint_parser: Optional[ConstraintParser]
        :param N_iterations: The number of iterations for the algorithm. Defaults to 1.
        :type N_iterations: int
        :param discount_factor: The factor for discounting the value of future costs. Defaults to 1.0.
        :type discount_factor: float
        :param policy_model: The model parameterizing the policy. Defaults to None. If `None` then `ModelWeightContainer` is used.
        :type policy_model: Optional[Model]
        :param critic_model:  The model parameterizing the critic. Defaults to None. If `None` then diagonal quadratic form is used.
        :type critic_model: Optional[Model]
        :param critic_regularization_param: The regularization parameter for the critic. Defaults to 0.
        :type critic_regularization_param: float

        :return: None
        :rtype: None
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

        :param running_objective: The objective function to assess the costs over the prediction horizon.
        :type running_objective: RunningObjective
        :param simulator: The environment simulation for applying and testing the agent.
        :type simulator: Simulator
        :param prediction_horizon: The number of steps into the future over which predictions are made.
        :type prediction_horizon: int
        :param sampling_time: The time step interval for scenario.
        :type sampling_time: float
        :param n_epochs: The number of training epochs.
        :type n_epochs: int
        :param opt_method_kwargs: Additional keyword arguments for the optimization method.
        :type opt_method_kwargs: Dict[str, Any]
        :param opt_method: The optimization method to use. Defaults to torch.optim.Adam.
        :type opt_method: Type[torch.optim.Optimizer]
        :param predictor: The predictor object to use. Defaults to None.
        :type predictor: Optional[Predictor]
        :param model: The neural network model parameterizing the policy. Defaults to None. If `None` then `ModelWeightContainerTorch` is used, but you can implement any neural network you want.
        :type model: Optional[ModelNN]
        :param observer: Object responsible for state estimation from observations. Defaults to None.
        :type observer: Optional[Observer]
        :param constraint_parser: The mechanism for enforcing operational constraints. Defaults to None.
        :type constraint_parser: Optional[ConstraintParser]
        :param discount_factor: The discount factor for future costs. Defaults to 1.0.
        :type discount_factor: float

        :return: None
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
                model=ModelWeightContainerTorch(
                    dim_weights=(prediction_horizon + 1, system.dim_inputs),
                    output_bounds=system.action_bounds,
                )
                if model is None
                else model,
                constraint_parser=constraint_parser,
                system=system,
                running_objective=running_objective,
                prediction_horizon=prediction_horizon,
                algorithm="mpc",
                critic=CriticTrivial(),
                predictor=predictor
                if predictor is not None
                else EulerPredictor(system=system, pred_step_size=sampling_time),
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
        model=ModelWeightContainerTorch(
            dim_weights=(
                prediction_horizon + (1 if algorithm_name != "rpv" else 0),
                system.dim_inputs,
            ),
            output_bounds=system.action_bounds,
            output_bounding_type="clip",
        )
        if policy_model is None
        else policy_model,
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
            is_value_function=critic_is_value_function,
            is_on_policy=True,
            is_same_critic=True,
            optimizer_config=TorchOptimizerConfig(
                critic_n_epochs,
                data_buffer_iter_bathes_kwargs=dict(
                    batch_sampler=EpisodicSampler,
                    dtype=torch.FloatTensor,
                ),
                opt_method_kwargs=critic_opt_method_kwargs,
                opt_method=critic_opt_method,
                is_reinstantiate_optimizer=is_reinstantiate_critic_optimizer,
            ),
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
        value_threshold=value_threshold,
        sampling_time=sampling_time,
        policy=policy_type(
            model=policy_model,
            system=system,
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
        epsilon: float = 0.2,
        discount_factor: float = 0.7,
        observer: Optional[Observer] = None,
        N_episodes: int = 2,
        N_iterations: int = 100,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    ):
        """Initialize the object with the given parameters.

        :param policy_model: The neural network model parameterizing the policy.
        :type policy_model: PerceptronWithTruncatedNormalNoise
        :param critic_model: The neural network model used for the value function approximation.
        :type critic_model: ModelPerceptron
        :param sampling_time: Time interval between two consecutive actions.
        :type sampling_time: float
        :param running_objective: A function that returns the scalar running cost or reward associated with an observation-action pair.
        :type running_objective: RunningObjective
        :param simulator: The simulation environment where the agent performs actions.
        :type simulator: Simulator
        :param critic_n_epochs: The number of epochs for which the critic is trained per iteration.
        :type critic_n_epochs: int
        :param policy_n_epochs: The number of epochs for which the policy is trained per iteration.
        :type policy_n_epochs: int
        :param critic_opt_method_kwargs: A dictionary of keyword arguments for the optimizer used for the critic.
        :type critic_opt_method_kwargs: Dict[str, Any]
        :param policy_opt_method_kwargs: A dictionary of keyword arguments for the optimizer used for the policy.
        :type policy_opt_method_kwargs: Dict[str, Any]
        :param critic_opt_method: The optimization algorithm class used for training the critic, e.g. torch.optim.Adam.
        :type critic_opt_method: Type[torch.optim.Optimizer]
        :param policy_opt_method: The optimization algorithm class used for training the policy, e.g. torch.optim.Adam.
        :type policy_opt_method: Type[torch.optim.Optimizer]
        :param running_objective_type: Specifies whether the running objective represents a 'cost' to minimize or a 'reward' to maximize.
        :type running_objective_type: str
        :param critic_td_n: The n-step temporal-difference parameter used for critic updates.
        :type critic_td_n: int
        :param epsilon: Clipping parameter that restricts the deviation of the new policy from the old policy.
        :type epsilon: float
        :param discount_factor: A factor applied to future rewards or costs to discount their value relative to immediate ones.
        :type discount_factor: float
        :param observer: Object responsible for state estimation from observations.
        :type observer: Optional[Observer]
        :param N_episodes: The number of episodes to run in every iteration.
        :type N_episodes: int
        :param N_iterations: The number of iterations to run in the scenario.
        :type N_iterations: int
        :param value_threshold: Threshold of the total objective to end an episode.
        :type value_threshold: float

        :raises AssertionError: If the `running_objective_type` is invalid.

        :return: None
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
                    epsilon=epsilon,
                    running_objective_type=running_objective_type,
                    sampling_time=sampling_time,
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
    ):
        """Initialize SDPG.

        :param policy_model: The policy network model that defines the policy architecture.
        :type policy_model: PerceptronWithTruncatedNormalNoise
        :param critic_model: The critic network model that defines the value function architecture.
        :type critic_model: ModelPerceptron
        :param sampling_time: The time step between agent actions in the environment.
        :type sampling_time: float
        :param running_objective: Function calculating the reward or cost at each time step when an action is taken.
        :type running_objective: RunningObjective
        :param simulator: The environment in which the agent operates, providing observation.
        :type simulator: Simulator
        :param critic_n_epochs: Number of epochs for which the critic network is trained at each iteration.
        :type critic_n_epochs: int
        :param policy_n_epochs: Number of epochs for which the policy network is trained at each iteration.
        :type policy_n_epochs: int
        :param critic_opt_method_kwargs: Parameters for the critic's optimization algorithm.
        :type critic_opt_method_kwargs: Dict[str, Any]
        :param policy_opt_method_kwargs: Parameters for the policy's optimization algorithm.
        :type policy_opt_method_kwargs: Dict[str, Any]
        :param critic_opt_method: Class of the optimizer to use for optimizing the critic,  defaults to torch.optim.Adam
        :type critic_opt_method: Type[torch.optim.Optimizer]
        :param policy_opt_method: Class of the optimizer to use for optimizing the policy, , defaults to torch.optim.Adam
        :type policy_opt_method: Type[torch.optim.Optimizer]
        :param critic_td_n: The number of steps to look ahead in the TD-target for the critic.
        :type critic_td_n: int
        :param discount_factor: The factor by which future rewards or costs are discounted relative to immediate running objectives.
        :type discount_factor: float
        :param observer: The observer object that estimates the state of the environment from observations.
        :type observer: Optional[Observer]
        :param N_episodes: The number of episodes per iteration.
        :type N_episodes: int
        :param N_iterations: The total number of iterations for training.
        :type N_iterations: int
        :param value_threshold: Threshold for the total objective that, once reached, stops the episode, defaults to np.inf.
        :type value_threshold: float
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

        :param policy_model: The policy network model that defines the policy architecture.
        :type policy_model: PerceptronWithTruncatedNormalNoise
        :param sampling_time: The time step between agent actions in the environment.
        :type sampling_time: float
        :param running_objective: Function calculating the reward or cost at each time step when an action is taken.
        :type running_objective: RunningObjective
        :param simulator: The environment in which the agent operates, providing state, observation.
        :type simulator: Simulator
        :param policy_opt_method_kwargs: The keyword arguments for the policy optimizer method.
        :type policy_opt_method_kwargs: Dict[str, Any]
        :param policy_opt_method: The policy optimizer method. Defaults to torch.optim.Adam.
        :type policy_opt_method: Type[torch.optim.Optimizer], optional
        :param n_epochs: The number of epochs used by the policy optimizer. Defaults to 1.
        :type n_epochs: int, optional
        :param discount_factor: The discount factor used by the RLScenario. Defaults to 1.0.
        :type discount_factor: float, optional
        :param observer: The observer object that estimates the state of the environment from observations. Defaults to None.
        :type observer: Optional[Observer], optional
        :param N_episodes: The number of episodes per iteration. Defaults to 4.
        :type N_episodes: int, optional
        :param N_iterations: The total number of iterations for training. Defaults to 100.
        :type N_iterations: int, optional
        :param is_with_baseline: Whether to use baseline as value (i.e. cumulative cost or reward) from previous iteration. Defaults to True.
        :type is_with_baseline: bool, optional
        :param is_do_not_let_the_past_distract_you: Whether to use tail total costs or not. Defaults to True.
        :type is_do_not_let_the_past_distract_you: bool, optional

        :return: None
        :rtype: None
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

        :param policy_model: The policy (actor) neural network model with input as state and output as action.
        :type policy_model: PerceptronWithTruncatedNormalNoise
        :param critic_model: The critic neural network model with input as state-action pair and output as Q-value estimate.
        :type critic_model: ModelPerceptron
        :param sampling_time: The time interval between each action taken by the policy.
        :type sampling_time: float
        :param running_objective: Th function calculating the running cost at each time step when an action is taken.
        :type running_objective: RunningObjective
        :param simulator: The environment simulator in which the agent operates.
        :type simulator: Simulator
        :param critic_n_epochs: The number of epochs for training the critic model during each optimization.
        :type critic_n_epochs: int
        :param policy_n_epochs: The number of epochs for training the policy model during each optimization.
        :type policy_n_epochs: int
        :param critic_opt_method_kwargs: Keyword arguments for the critic optimizer method.
        :type critic_opt_method_kwargs: Dict[str, Any]
        :param policy_opt_method_kwargs: Keyword arguments for the policy optimizer method.
        :type policy_opt_method_kwargs: Dict[str, Any]
        :param critic_opt_method: The optimizer class to be used for the critic. Defaults to torch.nn.Adam.
        :type critic_opt_method: Type[torch.optim.Optimizer]
        :param policy_opt_method: The optimizer class to be used for the policy. Defaults to torch.nn.Adam.
        :type policy_opt_method: Type[torch.optim.Optimizer]
        :param critic_td_n: The n-step return for temporal-difference learning for the critic estimator.
        :type critic_td_n: int
        :param discount_factor: The discount factor that weighs future costs lower compared to immediate costs.
        :type discount_factor: float
        :param observer: An observer object used for deriving state estimations from raw observations.
        :type observer: Optional[Observer]
        :param N_episodes: The number of episodes to be executed in each training iteration.
        :type N_episodes: int
        :param N_iterations: The total number of training iterations for the scenario.
        :type N_iterations: int
        :param value_threshold: The threshold for the total cumulative objective value that triggers the end of an episode.
        :type value_threshold: float
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
                is_reinstantiate_policy_optimizer=False,
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