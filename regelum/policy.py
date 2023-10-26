"""Module contains policies, i.e., entities that directly calculate actions. Policies are inegrated into controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""


import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional

from .__utilities import rc

from .predictor import Predictor
from .model import ModelNN, Model, ModelWeightContainer, ModelWeightContainerTorch
from .critic import Critic, CriticTrivial
from .system import System, ComposedSystem
from .optimizable.optimizers import Optimizable, OptimizerConfig
from .data_buffers.data_buffer import DataBuffer
from .constraint_parser import ConstraintParser, ConstraintParserTrivial
from .objective import (
    RunningObjective,
    reinforce_objective,
    sdpg_objective,
    ddpg_objective,
    mpc_objective,
    rpo_objective,
    sql_objective,
    rql_objective,
    ppo_objective,
)
from typing import List

try:
    import torch
except ImportError:
    from unittest.mock import MagicMock

    torch = MagicMock()


class Policy(Optimizable, ABC):
    """Class of policies.

    These are to be passed to a `controller`.
    """

    def __init__(
        self,
        model: Union[Model, ModelNN],
        system: Union[System, ComposedSystem] = None,
        action_bounds: Union[list, np.ndarray, None] = None,
        action_init=None,
        optimizer_config: Optional[OptimizerConfig] = None,
        discount_factor: Optional[float] = 1.0,
        epsilon_random: bool = False,
        epsilon_random_parameter: float = 0.0,
    ):
        """Initialize an instance of Policy class.

        :param model: Policy model.
        :type model: Union[Model, ModelNN]
        :param system: Agent environment, defaults to None
        :type system: Union[System, ComposedSystem], optional
        :param predictor: _description_, defaults to None
        :type predictor: Optional[Predictor], optional
        :param action_bounds: Bounds for the action., defaults to None
        :type action_bounds: Union[list, np.ndarray, None], optional
        :param action_init: _description_, defaults to None
        :type action_init: _type_, optional
        :param optimizer_config: Configuration of the optimization procedure, defaults to None
        :type optimizer_config: Optional[OptimizerConfig], optional
        :param discount_factor: _description_, defaults to 1.0
        :type discount_factor: Optional[float], optional
        :param epsilon_random: _description_, defaults to False
        :type epsilon_random: bool, optional
        :param epsilon_random_parameter: _description_, defaults to 0.0
        :type epsilon_random_parameter: float, optional
        """
        self.system = system
        self.model = model

        self.dim_action = self.system.dim_inputs
        self.dim_observation = self.system.dim_observation

        self.discount_factor = discount_factor if discount_factor is not None else 1.0
        self.epsilon_random = epsilon_random
        self.epsilon_random_parameter = epsilon_random_parameter

        super().__init__(optimizer_config=optimizer_config)
        (
            self.action_bounds,
            self.action_initial_guess,
            self.action_min,
            self.action_max,
        ) = self.handle_bounds(action_bounds, self.dim_action, 0)
        self.action_old = self.action_init = (
            self.action_initial_guess[: self.dim_action]
            if action_init is None
            else action_init
        )
        self.action = (
            self.action_initial_guess[: self.dim_action]
            if action_init is None
            else action_init
        )

    def __call__(self, observation):
        return self.get_action(observation)

    @property
    @abstractmethod
    def data_buffer_objective_keys(self) -> List[str]:
        pass

    @property
    def weights(self):
        """Get the weights of the policy model."""
        return self.model.weights

    def receive_observation(self, observation):
        """Update the current observation of the policy.

        :param observation: The current observation.
        :type observation: numpy array.
        """
        self.observation = observation

    def receive_state(self, state):
        """Update the current observation of the policy.

        :param observation: The current observation.
        :type observation: numpy array
        """
        self.state = state

    def set_action(self, action):
        """Set the current action of the policy.

        :param action: The current action.
        :type action: numpy array
        """
        self.action_old = self.action
        self.action = action

    def update_action(self, observation=None):
        """Update the current action of the policy.

        :param observation: The current observation. If not provided, the previously received observation will be used.
        :type observation: numpy array, optional
        """
        self.action_old = self.action

        if observation is None:
            observation = self.observation

        if self.epsilon_random:
            toss = np.random.choice(
                2,
                1,
                p=[
                    1 - self.epsilon_random_parameter,
                    self.epsilon_random_parameter,
                ],
            )
            is_exploration = bool(toss)

            if is_exploration:
                self.action = np.array(
                    [
                        np.random.uniform(self.action_min[k], self.action_max[k])
                        for k in range(len(self.action))
                    ]
                )
            else:
                self.action = self.get_action(observation)
        else:
            self.action = self.get_action(observation)

        return self.action

    def get_action(self, observation):
        if isinstance(self.model, ModelNN):
            action = self.model(observation).detach().numpy()
        else:
            action = self.model(observation)
        return action

    def update_weights(self, weights=None):
        """Update the weights of the model of the policy.

        :param weights: The weights to update the model with. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        self.model.update_weights(weights)

    def cache_weights(self, weights=None):
        """Cache the current weights of the model of the policy.

        :param weights: The weights to cache. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        self.model.cache_weights(weights)

    def update_and_cache_weights(self, weights=None):
        """Update and cache the weights of the model of the policy.

        :param weights: The weights to update and cache. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        self.model.update_and_cache_weights(weights)

    def restore_weights(self):
        """Restore the previously cached weights of the model of thepolicy."""
        self.model.restore_weights()
        self.set_action(self.action_old)

    def reset(self):
        """Reset the policy to its initial state."""
        self.action_old = self.action_initial_guess[: self.dim_action]
        self.action = self.action_initial_guess[: self.dim_action]


class PolicyGradient(Policy, ABC):
    """Base Class for policy gradient methods."""

    def __init__(
        self,
        model: ModelNN,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        discount_factor: float = 1.0,
        device: str = "cpu",
        critic: Critic = None,
    ):
        """Instantiate PolicyGradient base class.

        :param model: Policy model object.
        :type model: ModelNN
        :param system: Agent environment.
        :type system: Union[System, ComposedSystem]
        :param action_bounds: Action bounds.
        :type action_bounds: Union[list, np.ndarray, None]
        :param optimizer_config: Configuration of the optimizing procedure.
        :type optimizer_config: OptimizerConfig
        :param batch_keys: Keys for objective function.
        :type batch_keys: List[str]
        :param discount_factor: Discount factor for future running objectives, defaults to 1.0
        :type discount_factor: float, optional
        :param device: Device to proceed the optimization on, defaults to "cpu"
        :type device: str, optional
        :param critic: Critic object, defaults to None
        :type critic: Critic, optional
        """
        Policy.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            discount_factor=discount_factor,
            optimizer_config=optimizer_config,
        )
        self.device = torch.device(device)
        self.critic = critic

        self.N_episodes: int

    @abstractmethod
    def update_action(self, observation):
        pass

    def update_data_buffer(self, data_buffer: DataBuffer):
        pass

    def optimize_on_event(self, data_buffer: DataBuffer):
        # Send to device before optimization
        if self.critic is not None:
            self.critic.model = self.critic.model.to(self.device)
        self.model = self.model.to(self.device)
        self.N_episodes = len(np.unique(data_buffer.data["episode_id"]))
        self.update_data_buffer(data_buffer)

        self.optimize(
            **data_buffer.get_optimization_kwargs(
                keys=self.data_buffer_objective_keys(),
                optimizer_config=self.optimizer_config,
            )
        )

        # Send back to cpu after optimization
        if self.critic is not None:
            self.critic.model = self.critic.model.to(torch.device("cpu"))
        self.model = self.model.to(torch.device("cpu"))

    def initialize_optimization_procedure(self):
        self.objective_inputs = [
            self.create_variable(name=variable_name, is_constant=True)
            for variable_name in self.data_buffer_objective_keys()
        ]
        self.policy_weights = self.create_variable(
            name="policy_weights", like=self.model.named_parameters
        )
        self.register_objective(
            self.objective_function, variables=self.objective_inputs
        )

    @abstractmethod
    def objective_function(self, **kwargs):
        pass


class Reinforce(PolicyGradient):
    """The Reinforce class extends the PolicyGradient class and implements the REINFORCE algorithm."""

    def __init__(
        self,
        model: ModelNN,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        discount_factor: float = 1.0,
        device: str = "cpu",
        is_with_baseline: bool = True,
        is_do_not_let_the_past_distract_you: bool = False,
    ):
        """Instantiate Reinforce class.

        :param model: Policy model.
        :type model: ModelNN
        :param system: Agent environment.
        :type system: Union[System, ComposedSystem]
        :param action_bounds: Action bounds for the Agent.
        :type action_bounds: Union[list, np.ndarray, None]
        :param discount_factor: Discount factor for discounting future running objectives, defaults to 1.0
        :type discount_factor: float, optional
        :param device: Device for gradient step optimization, defaults to "cpu"
        :type device: str, optional
        :param is_with_baseline: Whether to use baseline in surrogate objective. Baseline is taken as total objective from the last iteration, defaults to True
        :type is_with_baseline: bool, optional
        :param is_do_not_let_the_past_distract_you: Whether to use tail total objectives in surrogate objective or not, defaults to False
        :type is_do_not_let_the_past_distract_you: bool, optional
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
            device=device,
        )
        self.is_with_baseline = is_with_baseline
        self.is_do_not_let_the_past_distract_you = is_do_not_let_the_past_distract_you
        self.next_baseline = 0.0

        self.initialize_optimization_procedure()

    def update_action(self, observation):
        self.action_old = self.action
        with torch.no_grad():
            self.action = (
                self.model.sample(torch.FloatTensor(observation)).cpu().numpy()
            )

    def update_data_buffer(self, data_buffer: DataBuffer):
        data_buffer.update(
            {
                "total_objective": self.calculate_last_total_objectives(data_buffer),
            }
        )
        data_buffer.update(
            {"tail_total_objective": self.calculate_tail_total_objectives(data_buffer)}
        )
        data_buffer.update({"baseline": self.calculate_baseline(data_buffer)})

    def calculate_last_total_objectives(self, data_buffer: DataBuffer):
        data = data_buffer.to_pandas(keys=["episode_id", "current_total_objective"])
        data["episode_id"] = data["episode_id"].astype(int)
        data["current_total_objective"] = data["current_total_objective"].astype(float)
        return (
            data.groupby("episode_id")["current_total_objective"]
            .last()
            .loc[data["episode_id"]]
            .values.reshape(-1)
        )

    def calculate_tail_total_objectives(
        self,
        data_buffer: DataBuffer,
    ):
        data = data_buffer.to_pandas(keys=["episode_id", "current_total_objective"])
        data["episode_id"] = data["episode_id"].astype(int)
        data["current_total_objective"] = data["current_total_objective"].astype(float)

        groupby_episode_total_objectives = data.groupby(["episode_id"])[
            "current_total_objective"
        ]

        last_total_objectives = (
            groupby_episode_total_objectives.last()
            .loc[data["episode_id"]]
            .values.reshape(-1)
        )
        current_total_objectives_shifted = groupby_episode_total_objectives.shift(
            periods=1, fill_value=0.0
        ).values.reshape(-1)

        return last_total_objectives - current_total_objectives_shifted

    def calculate_baseline(self, data_buffer: DataBuffer):
        baseline = self.next_baseline
        self.next_baseline = np.mean(
            data_buffer.to_pandas(keys=["total_objective"]).values
        )
        return np.full(shape=len(data_buffer), fill_value=baseline)

    def data_buffer_objective_keys(self) -> List[str]:
        return [
            "observation",
            "action",
            "tail_total_objective",
            "total_objective",
            "baseline",
        ]

    def objective_function(
        self, observation, action, tail_total_objective, total_objective, baseline
    ):
        return reinforce_objective(
            policy_model=self.model,
            observations=observation,
            actions=action,
            tail_total_objectives=tail_total_objective,
            total_objectives=total_objective,
            baselines=baseline,
            is_with_baseline=self.is_with_baseline,
            is_do_not_let_the_past_distract_you=self.is_do_not_let_the_past_distract_you,
            device=self.device,
            N_episodes=self.N_episodes,
        )


class SDPG(PolicyGradient):
    """Policy for Stochastic Deep Policy Gradient (SDPG)."""

    def __init__(
        self,
        model: ModelNN,
        critic: Critic,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        discount_factor: float = 1.0,
        device: str = "cpu",
    ):
        """Instantiate SDPG class.

        :param model: Policy Model.
        :type model: ModelNN
        :param critic: Critic object that is optmized via temporal difference objective.
        :type critic: Critic
        :param system: Agent environment.
        :type system: Union[System, ComposedSystem]
        :param action_bounds: Action bounds for the Agent.
        :type action_bounds: Union[list, np.ndarray, None]
        :param optimizer_config: Configuration of the optimization procedure.
        :type optimizer_config: OptimizerConfig
        :param discount_factor: Discounting factor for discounting future running objectives, defaults to 1.0
        :type discount_factor: float, optional
        :param device: Device to proceed the optimization process, defaults to "cpu"
        :type device: str, optional
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            discount_factor=discount_factor,
            device=device,
            critic=critic,
            optimizer_config=optimizer_config,
        )
        self.initialize_optimization_procedure()

    def data_buffer_objective_keys(self) -> List[str]:
        return ["observation", "action", "timestamp", "episode_id", "running_objective"]

    def objective_function(
        self, observation, action, timestamp, episode_id, running_objective
    ):
        return sdpg_objective(
            policy_model=self.model,
            critic_model=self.critic.model,
            observations=observation,
            actions=action,
            timestamps=timestamp,
            device=self.device,
            discount_factor=self.discount_factor,
            N_episodes=self.N_episodes,
            episode_ids=episode_id.long(),
            running_objectives=running_objective,
        )

    def update_action(self, observation):
        self.action_old = self.action
        with torch.no_grad():
            self.action = (
                self.model.sample(torch.FloatTensor(observation)).cpu().numpy()
            )


class PPO(PolicyGradient):
    """Proximal Policy Optimization."""

    def __init__(
        self,
        model: ModelNN,
        critic: Critic,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        discount_factor: float = 1.0,
        device: str = "cpu",
        running_objective_type="cost",
        epsilon: float = 0.2,
    ):
        """Instantiate PPO policy class.

        :param model: Policy Model.
        :type model: ModelNN
        :param critic: Critic object that is optmized via temporal difference objective.
        :type critic: Critic
        :param system: Agent environment.
        :type system: Union[System, ComposedSystem]
        :param action_bounds: Action bounds for the Agent.
        :type action_bounds: Union[list, np.ndarray, None]
        :param optimizer_config: Configuration of the optimization procedure.
        :type optimizer_config: OptimizerConfig
        :param discount_factor: Discounting factor for discounting future running objectives, defaults to 1.0
        :type discount_factor: float, optional
        :param device: Device to proceed the optimization process, defaults to "cpu"
        :type device: str, optional
        :param epsilon: Epsilon parameter, defaults to 0.2
        :type epsilon: float, optional
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            discount_factor=discount_factor,
            device=device,
            critic=critic,
            optimizer_config=optimizer_config,
        )
        self.epsilon = epsilon
        self.running_objective_type = running_objective_type
        self.initialize_optimization_procedure()

    def data_buffer_objective_keys(self) -> List[str]:
        return [
            "observation",
            "action",
            "timestamp",
            "episode_id",
            "running_objective",
            "initial_log_probs",
        ]

    def objective_function(
        self,
        observation,
        action,
        timestamp,
        episode_id,
        running_objective,
        initial_log_probs,
    ):
        return ppo_objective(
            policy_model=self.model,
            critic_model=self.critic.model,
            observations=observation,
            actions=action,
            timestamps=timestamp,
            device=self.device,
            discount_factor=self.discount_factor,
            N_episodes=self.N_episodes,
            episode_ids=episode_id.long(),
            running_objectives=running_objective,
            initial_log_probs=initial_log_probs,
            epsilon=self.epsilon,
            running_objective_type=self.running_objective_type,
        )

    def update_data_buffer(self, data_buffer: DataBuffer):
        data_buffer.update(
            {
                "initial_log_probs": self.model.log_pdf(
                    torch.FloatTensor(np.vstack(data_buffer.data["observation"])),
                    torch.FloatTensor(np.vstack(data_buffer.data["action"])),
                )
                .detach()
                .cpu()
                .numpy()
            }
        )

    def update_action(self, observation):
        self.action_old = self.action
        with torch.no_grad():
            self.action = (
                self.model.sample(torch.FloatTensor(observation)).cpu().numpy()
            )


class DDPG(PolicyGradient):
    """Policy for Deterministic Deep Policy Gradient (DDPG)."""

    def __init__(
        self,
        model: ModelNN,
        critic: Critic,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        discount_factor: float = 1.0,
        device: str = "cpu",
    ):
        """Instantiate DDPG class.

        :param model: Policy Model.
        :type model: ModelNN
        :param critic: Critic object that is optmized via temporal difference objective.
        :type critic: Critic
        :param system: Agent environment.
        :type system: Union[System, ComposedSystem]
        :param action_bounds: Action bounds for the Agent.
        :type action_bounds: Union[list, np.ndarray, None]
        :param optimizer_config: Configuration of the optimization procedure.
        :type optimizer_config: OptimizerConfig
        :param discount_factor: Discounting factor for discounting future running objectives, defaults to 1.0
        :type discount_factor: float, optional
        :param device: Device to proceed the optimization process, defaults to "cpu"
        :type device: str, optional
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            discount_factor=discount_factor,
            device=device,
            critic=critic,
            optimizer_config=optimizer_config,
        )
        self.initialize_optimization_procedure()

    def data_buffer_objective_keys(self) -> List[str]:
        return ["observation"]

    def objective_function(self, observation):
        return ddpg_objective(
            policy_model=self.model,
            critic_model=self.critic.model,
            observations=observation,
            device=self.device,
        )

    def update_action(self, observation):
        self.action_old = self.action
        with torch.no_grad():
            self.action = (
                self.model.sample(torch.FloatTensor(observation)).cpu().numpy()
            )


class RLPolicy(Policy):
    """Class for with Predictive Control algorithms."""

    def __init__(
        self,
        model: Union[ModelNN, Model],
        critic: Critic,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        predictor: Optional[Predictor] = None,
        prediction_horizon: Optional[int] = None,
        running_objective: Optional[RunningObjective] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        discount_factor: float = 1.0,
        device: str = "cpu",
        epsilon_random: bool = False,
        epsilon_random_parameter: float = 0.0,
        algorithm: str = "mpc",
    ):
        """Initialize an instance of RLPolicy class.

        :param model: Model for predictive policy
        :type model: Union[ModelNN, Model]
        :param critic: Critic for predictive policy (Can be used either Value or Action-Value critic variants)
        :type critic: Critic
        :param system:  a.k.a. environment. A class that represents the environment and contains dimensions of action and state space.
        :type system: Union[System, ComposedSystem]
        :param action_bounds: Bounds of actions represented by a list or numpy array, where first column is a minimal action and second column is a maximal action
        :type action_bounds: Union[list, np.ndarray, None]
        :param optimizer_config: A config for Optimizable
        :type optimizer_config: OptimizerConfig
        :param predictor: Predictor utilizing by policy to obtain a sequence of predictions, defaults to None.
        :type predictor: Optional[Predictor], optional
        :param prediction_horizon: length of predicted state sequence (needed for Predictior), defaults to None
        :type prediction_horizon: Optional[int], optional
        :param running_objective: Running objective of the control problem, defaults to None.
        :type running_objective: Optional[RunningObjective], optional
        :param discount_factor: Discounting factor for control problem. Defaults to 1.0
        :type discount_factor: float, optional
        :param device: Keyword argument specifying the device for torch (tensor) optimization. Defaults to "cpu"
        :type device: str, optional
        :param epsilon_random: If set to True, policy becomes epsilon-greedy with probability epsilon_random_parameter. Defaults to False
        :type epsilon_random: bool, optional
        :param epsilon_random_parameter: Probability of taking a random action during epsilon-greedy policy update phase. Defaults to 0.0
        :type epsilon_random_parameter: float, optional
        :param algorithm: Specifying the algorithm to which policy belongs and which objective function to optimize. Defaults to "mpc".
        :type algorithm: str, optional
        """
        Policy.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
        )

        self.action_bounds = action_bounds
        self.predictor = predictor
        self.prediction_horizon = prediction_horizon
        self.critic = critic
        self.device = device
        self.epsilon_random = epsilon_random
        self.epsilon_random_parameter = epsilon_random_parameter
        self.running_objective = running_objective
        self.algorithm = algorithm
        self.constraint_parser = (
            constraint_parser
            if constraint_parser is not None
            else ConstraintParserTrivial()
        )
        self.initialize_optimization_procedure()
        self.initial_guess = None

    def data_buffer_objective_keys(self) -> List[str]:
        return ["observation"]

    def initialize_optimization_procedure(self):
        objective_variables = []
        self.observation_variable = self.create_variable(
            1, self.system.dim_observation, name="observation", is_constant=True
        )
        objective_variables.append(self.observation_variable)
        self.policy_model_weights = self.create_variable(
            name="policy_model_weights", like=self.model.named_parameters
        )
        if isinstance(self.model, ModelWeightContainer):
            (
                self.action_bounds_tiled,
                self.action_initial_guess,
                self.action_min,
                self.action_max,
            ) = self.handle_bounds(
                self.action_bounds,
                self.dim_action,
                tile_parameter=self.model.weights.shape[0],
            )
            self.register_bounds(self.policy_model_weights, self.action_bounds_tiled)

        objective_variables.append(self.policy_model_weights)
        if not isinstance(self.critic, CriticTrivial):
            self.critic_weights = self.create_variable(
                name="critic_weights",
                like=self.critic.model.named_parameters,
                is_constant=True,
            )
            objective_variables.append(self.critic_weights)

        self.register_objective(
            self.objective_function,
            variables=objective_variables,
        )

        if len(list(self.constraint_parser.constraint_parameters())) > 0:
            if self.algorithm == "rpo":
                is_predict_last = True
            else:
                is_predict_last = False

            self.predicted_states_var = self.create_variable(
                self.prediction_horizon,
                self.system.dim_state,
                name="predicted_states",
                is_nested_function=True,
                nested_variables=[self.policy_model_weights],
            )
            self.connect_source(
                connect_to=self.predicted_states_var,
                func=self.predictor.predict_state_sequence_from_model,
                prediction_horizon=self.prediction_horizon,
                state=self.observation_variable,
                model=self.model,
                model_weights=self.policy_model_weights,
                is_predict_last=is_predict_last,
                return_predicted_states_only=True,
            )
            self.constraint_parameters = [
                self.create_variable(
                    *parameter.dims, name=parameter.name, is_constant=True
                ).with_data(parameter.data)
                for parameter in self.constraint_parser
            ]
            self.register_constraint(
                self.constraint_parser.constraint_function,
                variables=[self.predicted_states_var, *self.constraint_parameters],
            )

    def objective_function(
        self, observation, policy_model_weights, critic_weights=None
    ):
        if self.algorithm == "mpc":
            return mpc_objective(
                observation=observation,
                policy_model_weights=policy_model_weights,
                predictor=self.predictor,
                running_objective=self.running_objective,
                model=self.model,
                prediction_horizon=self.prediction_horizon,
                discount_factor=self.discount_factor,
            )
        elif self.algorithm == "rpo":
            return rpo_objective(
                observation=observation,
                policy_model_weights=policy_model_weights,
                predictor=self.predictor,
                running_objective=self.running_objective,
                model=self.model,
                prediction_horizon=self.prediction_horizon,
                discount_factor=self.discount_factor,
                critic=self.critic,
                critic_weights=critic_weights,
            )
        elif self.algorithm == "sql":
            return sql_objective(
                observation=observation,
                policy_model_weights=policy_model_weights,
                predictor=self.predictor,
                model=self.model,
                prediction_horizon=self.prediction_horizon,
                critic=self.critic,
                critic_weights=critic_weights,
            )
        elif self.algorithm == "rql":
            return rql_objective(
                observation=observation,
                policy_model_weights=policy_model_weights,
                predictor=self.predictor,
                running_objective=self.running_objective,
                model=self.model,
                prediction_horizon=self.prediction_horizon,
                critic=self.critic,
                critic_weights=critic_weights,
                discount_factor=self.discount_factor,
            )
        elif self.algorithm == "greedy":
            return rc.mean(
                self.critic(
                    observation,
                    self.model(observation, weights=policy_model_weights),
                    weights=critic_weights,
                )
            )
        else:
            raise AssertionError("RLPolicy: Wrong algorithm name")

    def get_sequential_output(self, observation):
        if isinstance(self.model, (ModelWeightContainerTorch, ModelWeightContainer)):
            pass

    def optimize_on_event(self, data_buffer: DataBuffer):
        opt_kwargs = data_buffer.get_optimization_kwargs(
            keys=self.data_buffer_objective_keys(),
            optimizer_config=self.optimizer_config,
        )

        if opt_kwargs is not None:
            if self.kind == "symbolic":
                result = (
                    self.optimize(
                        **opt_kwargs,
                        policy_model_weights=self.get_initial_guess(),
                        tol=1e-8,
                    )
                    if self.critic.weights is None
                    else self.optimize(
                        **opt_kwargs,
                        policy_model_weights=self.get_initial_guess(),
                        critic_weights=self.critic.weights,
                        tol=1e-8,
                    )
                )
                self.update_weights(result["policy_model_weights"])
            elif self.kind == "tensor":
                self.optimize(**opt_kwargs)

    def get_initial_guess(self):
        return self.model.weights


class CALFLegacy(RLPolicy):
    """Do not use it. Do not import it."""

    def __init__(
        self,
        safe_controller,
        *args,
        **kwargs,
    ):
        """Initialize thepolicy with a safe controller, and optional arguments for constraint handling, penalty term, andpolicy regularization.

        :param safe_controller: controller used to compute a safe action in case the optimization is rejected
        :type safe_controller: Controller
        :param policy_constraints_on: whether to use the CALF constraints in the optimization
        :type policy_constraints_on: bool
        :param penalty_param: penalty term for the optimization objective
        :type penalty_param: float
        :param policy_regularization_param: regularization term for thepolicy weights
        :type policy_regularization_param: float
        """
        super().__init__(*args, **kwargs)
        self.safe_controller = safe_controller
        self.penalty_param = penalty_param
        self.policy_regularization_param = policy_regularization_param
        self.predictive_constraint_violations = []
        self.intrinsic_constraints = (
            [
                self.CALF_decay_constraint_for_policy,
                # self.CALF_decay_constraint_for_policy_same_critic
            ]
            if policy_constraints_on
            else []
        )
        self.weights_acceptance_status = False
        safe_action = self.safe_controller.compute_action(
            self.state_init, self.critic.observation_last_good
        )
        self.action_init = self.action = safe_action
        self.model.update_and_cache_weights(safe_action)

    def CALF_decay_constraint_for_policy(self, weights=None):
        """Constraint for the policy optimization, ensuring that the critic value will not decrease by less than the required decay rate.

        :param weights:policy weights to be evaluated
        :type weights: numpy.ndarray
        :return: difference between the predicted critic value and the current critic value, plus the sampling time times the required decay rate
        :rtype: float
        """
        action = self.model(self.observation, weights=weights)

        self.predicted_observation = predicted_observation = self.predictor.predict(
            self.observation, action
        )
        observation_last_good = self.critic.observation_last_good

        self.critic_next = self.critic(predicted_observation)
        self.critic_current = self.critic(
            observation_last_good, use_stored_weights=True
        )

        self.predictive_constraint_violation = (
            self.critic_next
            - self.critic_current
            + self.critic.sampling_time * self.critic.safe_decay_param
        )

        return self.predictive_constraint_violation

    def CALF_decay_constraint_for_policy_same_critic(self, weights=None):
        """Calculate the predictive constraint violation for the CALF.

        This function calculates the violation of the "CALF decay constraint" which is used to ensure that the critic's value function
        (as a Lyapunov function) decreases over time. This helps to guarantee that the system remains stable.

        :param weights: (array) Weights for thepolicy model.
        :type weights: np.ndarray
        :return: (float) Predictive constraint violation.
        :rtype: float
        """
        action = self.model(self.observation, weights=weights)

        predicted_observation = self.predictor.predict(self.observation, action)
        observation_last_good = self.critic.observation_last_good

        self.predictive_constraint_violation = (
            self.critic(predicted_observation)
            - self.critic(observation_last_good)
            + self.critic.sampling_time * self.critic.safe_decay_param
        )

        return self.predictive_constraint_violation
