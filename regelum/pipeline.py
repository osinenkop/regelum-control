"""Contains high-level structures of pipelines (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import casadi
import torch

from .__utilities import rg, Clock, AwaitedParameter
from regelum import RegelumBase
from .policy import Policy, RLPolicy, PPO, Reinforce, SDPG, DDPG
from .critic import Critic, CriticCALF, CriticTrivial
from typing import Optional, Union, Type, Dict, List, Any
from .objective import RunningObjective
from .data_buffers import DataBuffer
from .event import Event
from . import OptStatus
from .simulator import Simulator
from .constraint_parser import ConstraintParser, ConstraintParserTrivial
from .observer import Observer, ObserverTrivial
from .model import (
    ModelNN,
    ModelPerceptron,
    ModelWeightContainer,
    ModelWeightContainerTorch,
    PerceptronWithTruncatedNormalNoise,
)
from .predictor import Predictor, EulerPredictor
from regelum.optimizable import OptimizerConfig
from regelum.data_buffers.batch_sampler import RollingBatchSampler, EpisodicSampler


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


class Pipeline(RegelumBase):
    """Pipeline orchestrator.

    A Pipeline orchestrates the training and evaluation cycle of a reinforcement learning agent.
    It runs the simulation based on a given policy, collects observations, applies actions, and
    manages the overall simulation loop, including assessing the agent's performance.
    """

    def __init__(
        self,
        policy: Policy,
        simulator: Simulator,
        sampling_time: float = 0.1,
        action_bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
        running_objective: Optional[RunningObjective] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        value_threshold: float = np.inf,
        discount_factor: float = 1.0,
    ):
        """Initialize the Pipeline with the necessary components for running a reinforcement learning experiment.

        :param policy: Policy to generate actions based on observations.
        :param simulator: Simulator to interact with and collect data for training.
        :param sampling_time: Time interval between action updates.
        :param action_bounds: Boundaries for valid actions.
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
        self.values_of_episodes = []
        self.value_episodic_means = []
        self.action_bounds = np.array(action_bounds)

        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.current_scenario_status = "episode_continues"
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

                self.reload_pipeline()

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
        self.recent_values_of_episodes = self.values_of_episodes
        self.values_of_episodes = []

    def reset_episode(self):
        self.episode_counter += 1
        self.is_episode_ended = False

        return self.value

    def reset_simulation(self):
        self.current_scenario_status = "episode_continues"
        self.iteration_counter = 0
        self.episode_counter = 0

    @apply_callbacks()
    def reload_pipeline(self):
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


class RLPipeline(Pipeline):
    """Incorporates reinforcement learning algorithms.

    The RLPipeline incorporates reinforcement learning algorithms into the Pipeline framework,
    enabling iterative optimization of both policies and value functions as part of the agent's learning process.
    """

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
        action_bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
        max_data_buffer_size: Optional[int] = None,
        sampling_time: float = 0.1,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        value_threshold: float = np.inf,
    ):
        """Instantiate a RLPipeline object.

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
        :param data_buffer_nullify_event: moments for DataBuffer nullifying. Can be either 'compute_action' for online learning, or 'reset_episode' for optimizing after each episode, or 'reset_iteration' for optimizing after each iteration
        :type data_buffer_nullify_event: str
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
        Pipeline.__init__(
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


class CALFPipelineExPost(RLPipeline):
    """Pipeline for CALF algorithm."""

    def __init__(
        self,
        policy: Policy,
        critic: CriticCALF,
        safe_pipeline: Pipeline,
        running_objective,
        critic_optimization_event: str,
        policy_optimization_event: str,
        data_buffer_nullify_event: str,
        discount_factor=1,
        action_bounds=None,
        max_data_buffer_size: Optional[int] = None,
        sampling_time: float = 0.1,
    ):
        """Instantiate a CALFPipelineExPost object. The docstring will be completed in the next release.

        :param policy: Pol
        :type policy: Policy
        :param critic: _description_
        :type critic: Critic
        :param safe_pipeline: _description_
        :type safe_pipeline: Pipeline
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
            sampling_time=sampling_time,
        )
        self.safe_pipeline = safe_pipeline

    def invoke_safe_action(self, state, observation):
        self.policy.restore_weights()
        self.critic.restore_weights()
        action = self.safe_pipeline.compute_action(state, observation)
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

        if rg.norm_2(observation) > 0.5:
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


class MPCPipeline(RLPipeline):
    """Leverages the Model Predictive Control Pipeline.

    The MPCPipeline leverages the Model Predictive Control (MPC) approach within the reinforcement learning pipeline,
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
        :param sampling_time:  The time step interval for pipeline.
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
            data_buffer_nullify_event=Event.reset_episode,
            policy_optimization_event=Event.compute_action,
            action_bounds=system.action_bounds,
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
                optimizer_config=OptimizerConfig(
                    log_options={
                        "print_in": False,
                        "print_out": False,
                        "print_time": False,
                    },
                    config_options={
                        "data_buffer_sampling_method": "sample_last",
                        "data_buffer_sampling_kwargs": {"dtype": casadi.DM},
                    },
                    opt_method="ipopt",
                    opt_options={"print_level": 0},
                    kind="symbolic",
                ),
            ),
        )


class MPCTorchPipeline(RLPipeline):
    """MPCTorchPipeline encapsulates the model predictive control (MPC) approach using PyTorch optimization for reinforcement learning.

    It integrates a policy that employs model-based predictions over a specified horizon to optimize actions, taking into account the dynamic nature of the environment. The pipeline coordinates the interaction between the policy, model, critic, and environment.
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
        :param sampling_time: The time step interval for pipeline.
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
            N_episodes=1,
            N_iterations=1,
            data_buffer_nullify_event=Event.reset_episode,
            policy_optimization_event=Event.compute_action,
            action_bounds=system.action_bounds,
            critic=CriticTrivial(),
            running_objective=running_objective,
            observer=observer,
            sampling_time=sampling_time,
            policy=RLPolicy(
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
                optimizer_config=OptimizerConfig(
                    config_options={
                        "n_epochs": n_epochs,
                        "data_buffer_sampling_method": "iter_batches",
                        "data_buffer_sampling_kwargs": {
                            "dtype": torch.FloatTensor,
                            "mode": "backward",
                            "batch_size": 1,
                        },
                    },
                    opt_method=opt_method,
                    opt_options=opt_method_kwargs,
                    kind="tensor",
                ),
            ),
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
    pipeline_kwargs: Dict[str, Any] = None,
    is_use_critic_as_policy_kwarg: bool = True,
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
            optimizer_config=OptimizerConfig(
                config_options={
                    "n_epochs": critic_n_epochs,
                    "data_buffer_sampling_method": "iter_batches",
                    "data_buffer_sampling_kwargs": {
                        "batch_sampler": EpisodicSampler,
                        "dtype": torch.FloatTensor,
                    },
                    "is_reinstantiate_optimizer": is_reinstantiate_critic_optimizer,
                },
                opt_method=critic_opt_method,
                opt_options=critic_opt_method_kwargs,
                kind="tensor",
            ),
            **(critic_kwargs if critic_kwargs is not None else dict()),
        )
    else:
        critic = CriticTrivial()

    return dict(
        simulator=simulator,
        discount_factor=discount_factor,
        data_buffer_nullify_event=Event.reset_iteration,
        policy_optimization_event=Event.reset_iteration,
        critic_optimization_event=Event.reset_iteration,
        N_episodes=N_episodes,
        N_iterations=N_iterations,
        running_objective=running_objective,
        action_bounds=system.action_bounds,
        observer=observer,
        critic=critic,
        value_threshold=value_threshold,
        sampling_time=sampling_time,
        policy=policy_type(
            model=policy_model,
            system=system,
            discount_factor=discount_factor,
            optimizer_config=OptimizerConfig(
                config_options={
                    "n_epochs": policy_n_epochs,
                    "data_buffer_sampling_method": "iter_batches",
                    "data_buffer_sampling_kwargs": {
                        "batch_sampler": RollingBatchSampler,
                        "dtype": torch.FloatTensor,
                        "mode": "full",
                        "n_batches": 1,
                    },
                    "is_reinstantiate_optimizer": is_reinstantiate_policy_optimizer,
                },
                opt_method=policy_opt_method,
                opt_options=policy_opt_method_kwargs,
                kind="tensor",
            ),
            **(dict(critic=critic) if is_use_critic_as_policy_kwarg else dict()),
            **(policy_kwargs if policy_kwargs is not None else dict()),
        ),
        **(pipeline_kwargs if pipeline_kwargs is not None else dict()),
    )


class PPOPipeline(RLPipeline):
    """Pipeline for Proximal Polizy Optimization.

    PPOPipeline is a reinforcement learning pipeline implementing the Proximal Policy Optimization (PPO) algorithm.
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
        :param N_iterations: The number of iterations to run in the pipeline.
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
                policy_type=PPO,
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
            )
        )


class SDPGPipeline(RLPipeline):
    """Implements the Stochastic Deep Policy Gradient (SDPG) algorithm.

    SDPGPipeline implements the Stochastic Deep Policy Gradient (SDPG) algorithm,
    an off-policy actor-critic method using a deep policy to model stochastic policies and
    a deep network as the critic that approximates a value Function.

    The SDPG algorithm performs policy updates by maximizing the critic's output and updates
    the critic using temporal-difference learning. This pipeline orchestrates the training
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
    ):
        """Initialize SDPGPipeline.

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
                policy_type=SDPG,
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
            )
        )


class ReinforcePipeline(RLPipeline):
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
    ):
        """Initialize an RLPipeline object.

        :param policy_model: he policy network model that defines the policy architecture.
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
        :param discount_factor: The discount factor used by the RLPipeline. Defaults to 1.0.
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
                policy_type=Reinforce,
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
            ),
        )


class DDPGPipeline(RLPipeline):
    """Implements a pipeline for interacting with an environment using the Deep Deterministic Policy Gradients (DDPG) algorithm."""

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
    ):
        """Instantiate DDPG Pipeline.

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
        :param N_iterations: The total number of training iterations for the pipeline.
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
                policy_type=DDPG,
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
            )
        )


# Additional classes such as Policy may need to be defined or imported depending on the setup of your framework.
