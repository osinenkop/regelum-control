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
from .model import ModelNN, Model
from .critic import Critic
from .system import System, ComposedSystem
from .optimizable.optimizers import Optimizable, OptimizerConfig
from .data_buffers.data_buffer import DataBuffer
from .objective import reinforce_objective, sdpg_objective, ddpg_objective
from typing import List

try:
    import torch
except ImportError:
    from unittest.mock import MagicMock

    torch = MagicMock()


class Policy(Optimizable):
    """Class of policies.
    These are to be passed to a `controller`.
    An `objective` (a loss) as well as an `optimizer` are passed to an `policy` externally.
    """  # noqa: D205

    def __init__(
        self,
        model,
        system: Union[System, ComposedSystem] = None,
        predictor: Optional[Predictor] = None,
        action_bounds: Union[list, np.ndarray, None] = None,
        action_init=None,
        optimizer_config: Optional[OptimizerConfig] = None,
        discount_factor: Optional[float] = 1.0,
        epsilon_random: bool = False,
        epsilon_random_parameter: float = 0.0,
    ):
        """Initialize an policy.

        :param prediction_horizon: Number of time steps to look into the future.
        :type prediction_horizon: int
        :param dim_observation: Dimension of the observation.
        :type dim_observation: int
        :param dim_action: Dimension of the action.
        :type dim_action: int
        :param action_bounds: Bounds on the action.
        :type action_bounds: list or ndarray, optional
        :param predictor: Predictor object for generating predictions.
        :type predictor: Predictor, optional
        :param optimizer: Optimizer object for optimizing the action.
        :type optimizer: Optimizer, optional
        :param critic: Critic object for evaluating actions.
        :type critic: Critic, optional
        :param running_objective: Running objective object for recording
        the running objective.
        :type running_objective: RunningObjective, optional
        :param model: Model object to be used as reference by the Predictor
        and the Critic.
        :type model: Model, optional
        :param discount_factor: discount factor to be used in conjunction with
        the critic.
        :type discount_factor: float, optional
        """
        self.system = system
        self.model = model
        self.predictor = predictor
        if self.predictor is not None:
            self.prediction_horizon = self.predictor.prediction_horizon
        else:
            self.prediction_horizon = 0

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
        ) = self.handle_bounds(
            action_bounds, self.dim_action, tile_parameter=self.prediction_horizon
        )
        self.action_variable = self.create_variable(
            system.dim_inputs, name="action", is_constant=True
        )
        self.register_bounds(self.action_variable, self.action_bounds)
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
                action = np.array(
                    [
                        np.random.uniform(self.action_min[k], self.action_max[k])
                        for k in range(len(self.action))
                    ]
                )
            else:
                action = self.get_action(observation)
        else:
            action = self.get_action(observation)

        return action

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
        if weights is None:
            self.model.update_weights(self.optimized_weights)
        else:
            self.model.update_weights(weights)

    def cache_weights(self, weights=None):
        """Cache the current weights of the model of the policy.

        :param weights: The weights to cache. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        if weights is not None:
            self.model.cache_weights(weights)
        elif self.optimized_weights is not None:
            self.model.cache_weights(self.optimized_weights)
        else:
            raise ValueError("Nothing to cache")

    def update_and_cache_weights(self, weights=None):
        """Update and cache the weights of the model of the policy.

        :param weights: The weights to update and cache. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        self.update_weights(weights)
        self.cache_weights(weights)

    def restore_weights(self):
        """Restore the previously cached weights of the model of thepolicy."""
        self.model.restore_weights()
        self.set_action(self.action_old)

    def get_initial_guess(self, guess_from):
        final_count_of_actions = self.prediction_horizon + 1
        action_sequence = rc.rep_mat(guess_from, 1, final_count_of_actions)
        action_sequence = rc.reshape(
            action_sequence,
            [final_count_of_actions * self.dim_action],
        )
        return action_sequence

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
        batch_keys: List[str],
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
        self.batch_keys = batch_keys
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
            dataloader=data_buffer.iter_batches(
                batch_size=len(data_buffer),
                dtype=torch.FloatTensor,
                keys=self.batch_keys,
                mode="forward",
                n_batches=1,
            ),
        )

        # Send back to cpu after optimization
        if self.critic is not None:
            self.critic.model = self.critic.model.to(torch.device("cpu"))
        self.model = self.model.to(torch.device("cpu"))

    def initialize_optimization_procedure(self):
        self.objective_inputs = [
            self.create_variable(name=variable_name, is_constant=True)
            for variable_name in self.batch_keys
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
            batch_keys=[
                "observation",
                "action",
                "total_objective",
                "tail_total_objective",
                "baseline",
            ],
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
        groupby_episode_total_objectives = data_buffer.to_pandas(
            keys=["episode_id", "current_total_objective"]
        ).groupby(["episode_id"])["current_total_objective"]
        return (
            groupby_episode_total_objectives.last()
            - groupby_episode_total_objectives.shift(periods=1, fill_value=0.0)
        ).values.reshape(-1)

    def calculate_baseline(self, data_buffer: DataBuffer):
        baseline = self.next_baseline
        self.next_baseline = np.mean(data_buffer.to_pandas(keys=["total_objective"]))
        return np.full(shape=len(data_buffer), fill_value=baseline)

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
        :param discount_factor: Discount factor for discounting future running objectives, defaults to 1.0
        :type discount_factor: float, optional
        :param device: Device to proceed the optimization process, defaults to "cpu"
        :type device: str, optional
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            batch_keys=["observation", "action", "timestamp"],
            discount_factor=discount_factor,
            device=device,
            critic=critic,
            optimizer_config=optimizer_config,
        )
        self.initialize_optimization_procedure()

    def objective_function(self, observation, action, timestamp):
        return sdpg_objective(
            policy_model=self.model,
            critic_model=self.critic.model,
            observations=observation,
            actions=action,
            timestamps=timestamp,
            device=self.device,
            discount_factor=self.discount_factor,
            N_episodes=self.N_episodes,
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
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            batch_keys=["observation"],
            discount_factor=discount_factor,
            device=device,
            critic=critic,
            optimizer_config=optimizer_config,
        )
        self.initialize_optimization_procedure()

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
    def __init__(
        self,
        model: ModelNN,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        critic: Critic,
        discount_factor: float = 1.0,
        device: str = "cpu",
        epsilon_random: bool = False,
        epsilon_random_parameter: float = 0.0,
    ):
        Policy.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
        )
        self.critic = critic
        self.device = device
        self.epsilon_random = epsilon_random
        self.epsilon_random_parameter = epsilon_random_parameter
        self.initialize_optimization_procedure()

    def initialize_optimization_procedure(self):
        self.policy_weights = self.create_variable(
            name="policy_weights",
            is_constant=False,
            like=self.model.named_parameters,
        )
        self.policy_model_output = self.create_variable(
            name="policy_model_output", is_constant=True
        )
        self.observation = self.create_variable(name="observation", is_constant=True)
        self.connect_source(
            self.policy_model_output, func=self.model, source=self.observation
        )
        self.register_objective(
            self.objective_function,
            variables=[self.observation, self.policy_model_output],
        )

    def update_action(self, observation=None):
        self.action_old = self.action
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
                self.action = np.random.uniform(
                    self.action_bounds[:, 0], self.action_bounds[:, 1]
                )
            else:
                self.action = self.get_action(observation)
        else:
            self.action = self.get_action(observation)

        return self.action

    def objective_function(self, observation, policy_model_output):
        return self.critic.model(observation, policy_model_output).mean()

    def optimize_on_event(self, data_buffer: DataBuffer):
        self.optimize(
            dataloader=data_buffer.iter_batches(
                **self.optimizer_config.config_options.iter_batches_config
            )
        )


class RPOPolicy(RLPolicy):
    def __init__(
        self,
        model: ModelNN,
        critic: Critic,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        predictor,
        running_objective,
        discount_factor: float = 1.0,
        device: str = "cpu",
        epsilon_random: bool = False,
        epsilon_random_parameter: float = 0.0,
    ):
        RLPolicy.__init__(
            self,
            model=model,
            critic=critic,
            system=system,
            action_bounds=action_bounds,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
            device=device,
            epsilon_random=epsilon_random,
            epsilon_random_parameter=epsilon_random_parameter,
        )
        self.predictor = predictor
        self.running_objective = running_objective

    def objective_function(self, observation, policy_model_output):
        observation_next = self.predictor.predict(
            observation[-1], policy_model_output[-1]
        )
        return self.running_objective(
            observation[-1], policy_model_output[-1]
        ) + self.critic.model(observation_next)


class RPOPolicyCasadi(RLPolicy):
    def __init__(
        self,
        model: ModelNN,
        critic: Critic,
        system: Union[System, ComposedSystem],
        action_bounds: Union[list, np.ndarray, None],
        optimizer_config: OptimizerConfig,
        predictor,
        running_objective,
        discount_factor: float = 1.0,
        device: str = "cpu",
        epsilon_random: bool = False,
        epsilon_random_parameter: float = 0.0,
    ):
        RLPolicy.__init__(
            self,
            model=model,
            critic=critic,
            system=system,
            action_bounds=action_bounds,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
            device=device,
            epsilon_random=epsilon_random,
            epsilon_random_parameter=epsilon_random_parameter,
        )
        self.predictor = predictor
        self.running_objective = running_objective

    def objective_function(self, observation, policy_model_output):
        observation_next = self.predictor.predict(
            observation[-1], policy_model_output[-1]
        )
        return self.running_objective(
            observation[-1], policy_model_output[-1]
        ) + self.critic.model(observation_next)


class MPC(Policy):
    r"""Model-predictive control (MPC)policy.

    Optimizes the followingpolicy objective:
    :math:`J^a \\left( y_k| \\{u\\}_k^{N_a+1} \\right) = \\sum_{i=0}^{N_a} \\gamma^i r(y_{i|k}, u_{i|k})`.

    Notation:

    * :math:`y`: observation
    * :math:`u`: action
    * :math:`N_a`: prediction horizon
    * :math:`gamma`: discount factor
    * :math:`r`: running objective function
    * :math:`\\{\\bullet\\}_k^N`: sequence from index :math:`k` to index :math:`k+N-1`
    * :math:`\\bullet_{i|k}`: element in a sequence with index :math:`k+i-1`
    """

    def objective(
        self,
        action_sequence,
        observation,
    ):
        """Calculate thepolicy objective for the given action sequence and observation using Model Predictive Control (MPC).

        :param action_sequence: sequence of actions to be evaluated in the objective function
        :type action_sequence: numpy.ndarray
        :param observation: current observation
        :type observation: numpy.ndarray
        :return: thepolicy objective for the given action sequence
        :rtype: float
        """
        action_sequence_reshaped = rc.reshape(
            action_sequence, [self.prediction_horizon + 1, self.dim_action]
        ).T

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_sequence = rc.column_stack(
            (observation, observation_sequence_predicted)
        )
        policy_objective = 0
        for k in range(self.prediction_horizon + 1):
            policy_objective += self.discount_factor**k * self.running_objective(
                observation_sequence[:, k], action_sequence_reshaped[:, k]
            )
        return policy_objective


class MPCTerminal(Policy):
    r"""Model-predictive control (MPC)policy.

    Optimizes the followingpolicy objective:
    :math:`J^a \\left( y_k| \\{u\\}_k^{N_a+1} \\right) = \\sum_{i=0}^{N_a} \\gamma^i r(y_{i|k}, u_{i|k})`

    Notation:

    * :math:`y`: observation
    * :math:`u`: action
    * :math:`N_a`: prediction horizon
    * :math:`gamma`: discount factor
    * :math:`r`: running objective function
    * :math:`\\{\\bullet\\}_k^N`: sequence from index :math:`k` to index :math:`k+N-1`
    * :math:`\\bullet_{i|k}`: element in a sequence with index :math:`k+i-1`
    """

    def objective(
        self,
        action_sequence,
        observation,
    ):
        r"""Calculate the policy objective for the given action sequence and observation using Model Predictive Control (MPC).

        :param action_sequence: sequence of actions to be evaluated in the objective function
        :type action_sequence: numpy.ndarray
        :param observation: current observation
        :type observation: numpy.ndarray
        :return: thepolicy objective for the given action sequence
        :rtype: float
        """
        action_sequence_reshaped = rc.reshape(
            action_sequence, [self.prediction_horizon + 1, self.dim_output]
        ).T

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_sequence = rc.column_stack(
            (observation, observation_sequence_predicted)
        )
        policy_objective = 0

        policy_objective += self.running_objective(
            observation_sequence[:, -1], action_sequence_reshaped[:, -1]
        )
        return policy_objective


class SQL(Policy):
    r"""Staked Q-learning (SQL)policy.

    Optimizes the followingpolicy objective:
    :math:`J^a \\left( y_k| \\{u\\}_k^{N_a+1} \\right) = \\sum_{i=0}^{N_a} \\gamma^i Q(y_{i|k}, u_{i|k})`

    Notation:

    * :math:`y`: observation
    * :math:`u`: action
    * :math:`N_a`: prediction horizon
    * :math:`gamma`: discount factor
    * :math:`r`: running objective function
    * :math:`Q`: action-objective function (or its estimate)
    * :math:`\\{\\bullet\\}_k^N`: sequence from index :math:`k` to index :math:`k+N-1`
    * :math:`\\bullet_{i|k}`: element in a sequence with index :math:`k+i-1`
    """

    def objective(
        self,
        action_sequence,
        observation,
    ):
        """Calculate the policy objective for the given action sequence and observation using the stacked Q-learning (SQL) algorithm.

        :param action_sequence: numpy array of shape (prediction_horizon+1, dim_output) representing the sequence of actions to optimize
        :type action_sequence: numpy.ndarray
        :param observation: numpy array of shape (dim_output,) representing the current observation
        :type observation: numpy.ndarray
        :return:policy objective for the given action sequence and observation
        :rtype: float
        """
        action_sequence_reshaped = rc.reshape(
            action_sequence, [self.prediction_horizon + 1, self.dim_output]
        ).T

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_sequence = rc.column_stack(
            (
                observation,
                observation_sequence_predicted,
            )
        )

        policy_objective = 0

        for k in range(self.prediction_horizon + 1):
            action_objective = self.critic(
                observation_sequence[:, k],
                action_sequence_reshaped[:, k],
                use_stored_weights=True,
            )

            policy_objective += action_objective
        return policy_objective


class RQL(Policy):
    r"""Rollout Q-learning (RQL)policy.

    Optimizes the followingpolicy objective:

    :math:`J^a \\left( y_k| \\{u\\}_k^{N_a+1} \\right) = \\sum_{i=0}^{N_a-1} \\gamma^i r(y_{i|k}, u_{i|k}) + \\gamma^{N_a} Q(y_{N_a|k}, u_{N_a|k})`

    Notation:

    * :math:`y`: observation
    * :math:`u`: action
    * :math:`N_a`: prediction horizon
    * :math:`gamma`: discount factor
    * :math:`r`: running objective function
    * :math:`Q`: action-objective function (or its estimate)
    * :math:`\\{\\bullet\\}_k^N`: sequence from index :math:`k` to index :math:`k+N-1`
    * :math:`\\bullet_{i|k}`: element in a sequence with index :math:`k+i-1`
    """

    def objective(
        self,
        action_sequence,
        observation,
    ):
        """Calculate the policy objective for the given action sequence and observation using Rollout Q-learning (RQL).

        :param action_sequence: numpy array of shape (prediction_horizon+1, dim_output) representing the sequence of actions to optimize
        :type action_sequence: numpy.ndarray
        :param observation: numpy array of shape (dim_output,) representing the current observation
        :type observation: numpy.ndarray
        :return:policy objective for the given action sequence and observation
        :rtype: float
        """
        action_sequence_reshaped = rc.reshape(
            action_sequence, [self.prediction_horizon + 1, self.dim_output]
        ).T

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_sequence = rc.column_stack(
            (
                observation,
                observation_sequence_predicted,
            )
        )

        policy_objective = 0

        for k in range(self.prediction_horizon):
            policy_objective += self.discount_factor**k * self.running_objective(
                observation_sequence[:, k], action_sequence_reshaped[:, k]
            )

        policy_objective += self.critic(
            observation_sequence[:, -1],
            action_sequence_reshaped[:, -1],
            use_stored_weights=True,
        )

        return policy_objective


class RPO(Policy):
    r"""Running (objective) Plus Optimal (objective)policy.

    Policy minimizing the sum of the running objective and the optimal (or estimate thereof) objective of the next step.
    May be suitable for value iteration and policy iteration agents.
    Specifically, it optimizes the followingpolicy objective:

    :math:`J^a \\left( y_k| \\{u\\}_k \\right) =  r(y_{k}, u_{k}) + \\gamma J^*(y_{k})`

    Notation:

    * :math:`y`: observation
    * :math:`u`: action
    * :math:`\\gamma`: discount factor
    * :math:`r`: running objective function
    * :math:`J^*`: optimal objective function (or its estimate)
    """

    def objective(
        self,
        action_sequence,
        observation,
    ):
        """Calculate the policy objective for the given action sequence and observation using Running Plus Optimal (RPO).

        :param action_sequence: numpy array of shape (prediction_horizon+1, dim_input) representing the sequence of actions to optimize
        :type action_sequence: numpy.ndarray
        :param observation: numpy array of shape (dim_input,) representing the current observation
        :type observation: numpy.ndarray
        :return:policy objective for the given action sequence and observation
        :rtype: float
        """
        current_action = action_sequence[: self.dim_action]

        observation_predicted = self.predictor.predict(observation, current_action)

        running_objective_value = self.running_objective(observation, current_action)

        critic_of_observation = self.critic(observation_predicted)

        policy_objective = running_objective_value + critic_of_observation

        if self.intrinsic_constraints != [] and self.penalty_param > 0:
            for constraint in self.intrinsic_constraints:
                policy_objective += self.penalty_param * rc.penalty_function(
                    constraint(), penalty_coeff=1.0e-1
                )

        return policy_objective


class RPOWithRobustigyingTerm(RPO):
    """Class for RPO with robustigying term."""

    def __init__(self, *args, A=10, K=10, **kwargs):
        """Instatiate the class for RPO with robustigying term.

        TODO: add link to paper

        :param A: Robustifiend parameter 1, defaults to 10
        :type A: int, optional
        :param K: Robustifiend parameter 2, defaults to 10
        :type K: int, optional
        """
        super().__init__(*args, **kwargs)
        self.A = A
        self.K = K

    def update_action(self, observation=None):
        super().update_action(observation)
        self.action -= (
            self.K
            * rc.norm_2(observation) ** 2
            / (self.A + rc.norm_2(observation) ** 2)
        )


class CALF(RPO):
    """Policy using Critic As a Lyapunov Function (CALF) to constrain the optimization."""

    def __init__(
        self,
        safe_controller,
        *args,
        policy_constraints_on=True,
        penalty_param=0,
        policy_regularization_param=0,
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
        """Constraint for thepolicy optimization, ensuring that the critic value will not decrease by less than the required decay rate.

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


class CLF(CALF):
    """PolicyCLF is anpolicy class that aims to optimize the decay of a Control-Lyapunov function (CLF)."""

    def __init__(self, *args, **kwargs):
        """Initialize the policy with a safe controller, and optional arguments for constraint handling, penalty term, andpolicy regularization.

        :param safe_controller: object of class SafeController that provides a safe action if the current action would violate the safe set.
        :type safe_controller: SafeController
        """
        super().__init__(*args, **kwargs)
        self.intrinsic_constraints = []

    def objective(
        self,
        action,
        observation,
    ):
        """Compute the anticipated decay of the CLF.

        :param action: Action taken by thepolicy.
        :type action: ndarray
        :param observation: Observation of the system.
        :type observation: ndarray
        :return: Policy objective
        :rtype: float
        """
        observation_predicted = self.predictor.predict(observation, action)

        policy_objective = 0

        policy_objective += self.safe_controller.compute_LF(observation_predicted)

        return policy_objective


class Tabular(RPO):
    r"""Policy minimizing the sum of the running objective and the optimal (or estimate thereof) objective of the next step.

    May be suitable for value iteration and policy iteration agents.
    Specifically, it optimizes the followingpolicy objective:

    :math:`J^a \\left( y_k| \\{u\\}_k \\right) =  r(y_{k}, u_{k}) + \\gamma J^*(y_{k})`

    Notation:

    * :math:`y`: observation
    * :math:`u`: action
    * :math:`\\gamma`: discount factor
    * :math:`r`: running objective function
    * :math:`J^*`: optimal objective function (or its estimate)

    The action and state space are assumed discrete and finite.
    """

    def __init__(
        self,
        dim_world,
        predictor=None,
        optimizer=None,
        running_objective=None,
        model=None,
        action_space=None,
        critic=None,
        discount_factor=1,
        terminal_state=None,
    ):
        """Initialize anpolicyTabular object.

        :param dim_world: The dimensions of the world (i.e. the dimensions of the state space).
        :type dim_world: int
        :param predictor: An object that predicts the next state given an action and the current state.
        :type predictor: object
        :param optimizer: An object that optimizes thepolicy's objective function.
        :type optimizer: object
        :param running_objective: A function that returns a scalar representing the running objective for a given state and action.
        :type running_objective: function
        :param model: An object that computes an action given an observation and some weights.
        :type model: object
        :param action_space: An array of the possible actions.
        :type action_space: array
        :param critic: An object that computes the optimal objective function.
        :type critic: object
        :param discount_factor: The discount factor for the optimal objective function.
        :type discount_factor: float
        :param terminal_state: The terminal state of the world.
        :type terminal_state: object
        """
        self.dim_world = dim_world
        self.predictor = predictor
        self.critic = critic
        self.model = model
        self.running_objective = running_objective
        self.optimizer = optimizer
        self.action_space = action_space
        self.action_table = rc.zeros(dim_world)
        self.discount_factor = discount_factor
        self.terminal_state = terminal_state
        self.gradients = []

    def update(self):
        """Update the action table using the optimizer."""
        new_action_table = self.optimizer.optimize(self.objective, self.model.weights)

        self.model.update_and_cache_weights(new_action_table)

    def objective(
        self,
        action,
        observation,
    ):
        """Calculate the policy objective for a given action and observation.

        The policy objective is defined as the sum of the running objective and the optimal (or estimate thereof) objective of the next step.

        :param action: The action for which thepolicy objective is to be calculated.
        :type action: np.ndarray
        :param observation: The observation for which thepolicy objective is to be calculated.
        :type observation: np.ndarray
        :return: the policy objective.
        :rtype: float
        """
        if tuple(observation) == tuple(self.terminal_state):
            return 0

        observation_predicted = self.predictor.predict_sequence(observation, action)

        policy_objective = self.running_objective(
            observation, action
        ) + self.discount_factor * self.critic(observation_predicted)

        return policy_objective
