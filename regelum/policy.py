"""Module contains policies, i.e., entities that directly calculate actions. 

The Policy calculates actions based on observations from 
the environment and the current state of the model. 
It can be optimized using various optimization techniques, 
adjusting the model's parameters to improve the chosen actions over time.

Policies are integrated into scenarios, allowing them to operate within agents 
that interact with dynamic environments, like those found in reinforcement learning.

Remarks:

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional
import casadi as cs
from .utils import rg, AwaitedParameter
from regelum.typing import Weights
from .predictor import Predictor
from .model import ModelNN, Model, ModelWeightContainer
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
    rpv_objective,
    sql_objective,
    rql_objective,
    ppo_objective,
)
from typing import List, Tuple
from regelum.typing import RgArray
from typing import Any

import torch


class Policy(Optimizable, ABC):
    """Class of policies.

    Policy defines the strategy or rule by which an agent interacts with an environment.

    The Policy calculates actions based on observations from
    the environment and the current state of the model.
    It can be optimized using various optimization techniques,
    adjusting the model's parameters to improve the chosen actions over time.

    Policies are integrated into scenarios, allowing them to operate within agents
    that interact with dynamic environments, like those found in reinforcement learning.
    """

    def __init__(
        self,
        model: Union[Model, ModelNN] = None,
        system: Union[System, ComposedSystem] = None,
        action_bounds: Union[list, np.ndarray, None] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        discount_factor: Optional[float] = 1.0,
        epsilon_random_parameter: Optional[float] = None,
    ):
        """Initialize an instance of Policy class.

        Args:
            model: The model representing the policy's decision-making mechanism.
            system: System in which the policy will be deployed.
            action_bounds: Limits to the range of actions the policy can generate.
            optimizer_config: Configuration settings for the optimization procedure.
            discount_factor: Discount factor for the future running objectives.
            epsilon_random_parameter: Parameter that defines the randomness in action selection to encourage exploration.
        """
        self.system = system
        self.model = model

        self.dim_action = self.system.dim_inputs if self.system is not None else None
        self.dim_observation = (
            self.system.dim_observation if self.system is not None else None
        )

        self.discount_factor = discount_factor if discount_factor is not None else 1.0
        self.epsilon_random_parameter = epsilon_random_parameter
        if optimizer_config is not None:
            super().__init__(optimizer_config=optimizer_config)
        if self.dim_action is not None:
            (
                self.action_bounds,
                self.action_initial_guess,
                self.action_min,
                self.action_max,
            ) = self.handle_bounds(action_bounds, self.dim_action, 0)

        self.action = AwaitedParameter(
            "action", awaited_from=self.update_action.__name__
        )

    def __call__(self, observation: np.array) -> np.array:
        """Call the get_action method to calculate the action for the given observation.

        Args:
            observation: The observation based on which the action is to be calculated.

        Returns:
            The action calculated by the policy.
        """
        return self.get_action(observation)

    @property
    def action(self):
        """Return the action calculated by the policy."""
        return self._action

    @action.setter
    def action(self, value):
        """Get the action calculated by the policy."""
        self._action = value

    @action.getter
    def action(self):
        """Get the action calculated by the policy."""
        if isinstance(self._action, cs.DM):
            return self._action.full()
        elif isinstance(self._action, torch.Tensor):
            return self._action.detach().cpu().numpy()
        return self._action

    @property
    def data_buffer_objective_keys(self) -> Optional[List[str]]:
        """Provide the dictionary keys used for the objective function within data buffers.

        Returns:
            A list of keys used within the data buffers for objective function calculation.
        """
        pass

    @property
    def weights(self) -> Weights:
        """Get the weights of the policy model."""
        return self.model.weights

    def receive_observation(self, observation: np.array) -> None:
        """Update the current observation of the policy.

        Args:
            observation: The current observation.
        """
        self.observation = observation

    def receive_estimated_state(self, state: np.array) -> None:
        """Update the current observation of the policy.

        Args:
            state: The current state.
        """
        self.state = state

    def set_action(self, action: np.array):
        """Set the current action of the policy.

        Args:
            action: The current action.
        """
        self.action = action

    def update_action(self, observation: Optional[np.array] = None) -> np.array:
        """Update the current action of the policy based on the provided observation.

        This method uses the current model to compute a new action,
        possibly incorporating random elements for exploration purposes.
        If no observation is provided, the method uses the last received observation.

        Args:
            observation: The most recent observation received from the environment. If not
                provided, the previously received observation is used.
        """
        if observation is None:
            observation = self.observation

        if self.epsilon_random_parameter is not None:
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

    def get_action(self, observation: np.array) -> np.array:
        if isinstance(self.model, ModelNN):
            action = self.model(torch.FloatTensor(observation)).detach().cpu().numpy()
        else:
            action = self.model(observation)
        return action

    def update_weights(self, weights: Optional[Weights] = None) -> None:
        """Update the weights of the model of the policy.

        Args:
            weights (numpy array, optional): The weights to update the
                model with. If not provided, the previously optimized
                weights will be used.
        """
        self.model.update_weights(weights)

    def cache_weights(self, weights: Optional[Weights] = None) -> None:
        """Cache the current weights of the model of the policy.

        Args:
            weights: The weights to cache. If not provided, the current weights will be used.
        """
        self.model.cache_weights(weights)

    def update_and_cache_weights(self, weights: Optional[Weights] = None):
        """Update and cache the weights of the model of the policy.

        Args:
            weights: The weights to update and cache. If not provided, the previously optimized weights will be used.
        """
        self.model.update_and_cache_weights(weights)

    def restore_weights(self):
        """Restore the previously cached weights of the model of thepolicy."""
        self.model.restore_weights()


class PolicyGradient(Policy, ABC):
    """Base Class for policy gradient methods.

    Base class for policy gradient methods. Policy gradients methods
    optimize policies based on the gradient of some performance measure
    with respect to policy parameters. These methods typically involve
    estimating the gradient using a large number of samples (trajectories
    for which we have either complete values or advantage estimates).
    """

    def __init__(
        self,
        model: ModelNN,
        system: Union[System, ComposedSystem],
        optimizer_config: OptimizerConfig,
        discount_factor: float = 1.0,
        device: str = "cpu",
        critic: Critic = None,
    ):
        """Instantiate PolicyGradient base class.

        Args:
            model: The neural network representing the policy.
            system: The system or environment where the policy is deployed.
            optimizer_config: Configuration for the optimization process.
            discount_factor: The discount factor for future running objectives. Defaults to 1.0, indicating no discount.
            device: The device for computations, either 'cpu' or 'cuda:0', etc..
            critic: The critic object, could be either value or action-value based.

        Raises:
            NotImplementedError: If the method is used directly without a concrete implementation in a subclass.
        """
        Policy.__init__(
            self,
            model=model,
            system=system,
            discount_factor=discount_factor,
            optimizer_config=optimizer_config,
        )
        self.device = torch.device(device)
        self.critic = critic

        self.N_episodes: int

    def update_data_buffer(self, data_buffer: DataBuffer) -> None:
        """Add precomputed values to the data buffer.

        Args:
            data_buffer: Data buffer with experience replay.
        """

        pass

    def optimize(self, data_buffer: DataBuffer) -> None:
        """optimize the policy based on the given data buffer.

        Note:
            Concrete implementations of this method should follow the optimization
            procedure set by the `optimizer_config`.

        Args:
            data_buffer: Buffer storing interaction data with the environment used for optimizing the policy.
        """

        # Send to device before optimization
        if self.critic is not None:
            self.critic.model = self.critic.model.to(self.device)
        self.model = self.model.to(self.device)
        self.N_episodes = len(np.unique(data_buffer.data["episode_id"]))
        self.update_data_buffer(data_buffer)

        super().optimize(
            **data_buffer.get_optimization_kwargs(
                keys=self.data_buffer_objective_keys(),
                optimizer_config=self.optimizer_config,
            )
        )

        # Send back to cpu after optimization
        if self.critic is not None:
            self.critic.model = self.critic.model.to(torch.device("cpu"))
        self.model = self.model.to(torch.device("cpu"))

    def initialize_optimization_procedure(self) -> None:
        """initialize all parameters necessary to carry out the optimization procedure."""
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
    def objective_function(self, **kwargs: Any) -> torch.Tensor:
        """Abstract method defining the objective function to be optimized.

        Note:
            The objective function should be formulated based on kwargs which represent variables needed
            for computing the function such as observations, actions, estimated advantages, etc.

        Args:
            **kwargs: Additional keyword arguments that the objective function might need.

        Raises:
            NotImplementedError: If the method is used directly without concrete implementation in a subclass.
        """
        pass


class PolicyReinforce(PolicyGradient):
    """The Reinforce class extends the PolicyGradient class and implements the REINFORCE algorithm."""

    def __init__(
        self,
        model: ModelNN,
        system: Union[System, ComposedSystem],
        optimizer_config: OptimizerConfig,
        discount_factor: float = 1.0,
        device: str = "cpu",
        is_with_baseline: bool = True,
        is_do_not_let_the_past_distract_you: bool = True,
    ):
        """Instantiate Reinforce class.

        Args:
            model: The neural network-based policy model.
            system: The environment or system where the policy is deployed.
            optimizer_config: Settings for the associated optimization procedure.
            discount_factor: Discount factor for future running objectives, usually in the interval [0, 1].
            device: The device ('cpu' or 'cuda') to conduct the optimization process
            is_with_baseline (bool, optional): Indicates the use of a baseline to reduce variance in objective computation of gradients.
            is_do_not_let_the_past_distract_you: Whether to use tail values in objective.
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
            device=device,
        )
        self.is_with_baseline = is_with_baseline
        self.is_do_not_let_the_past_distract_you = is_do_not_let_the_past_distract_you
        self.next_baseline = 0.0

        self.initialize_optimization_procedure()

    def update_data_buffer(self, data_buffer: DataBuffer) -> None:
        """Prepares the data buffer for optimization by updating values such as `"value"`, `"tail_value"`, and `"baseline"`.

        Args:
            data_buffer: The data buffer containing trajectory information for optimization.

        Note:
            Does not return any values but updates the data buffer in place.

        """
        data_buffer.update(
            {
                "value": self.calculate_last_values(data_buffer),
            }
        )
        data_buffer.update({"tail_value": self.calculate_tail_values(data_buffer)})
        data_buffer.update({"baseline": self.calculate_baseline(data_buffer)})

    def calculate_last_values(self, data_buffer: DataBuffer) -> np.array:
        """Computes the last observed value in each trajectory and maps it to all steps within the same episode.

        Args:
            data_buffer: The data buffer containing trajectory information.

        Returns:
            A numpy array containing the last value of each trajectory for all steps.
        """

        data = data_buffer.to_pandas(keys=["episode_id", "current_value"])
        data["episode_id"] = data["episode_id"].astype(int)
        data["current_value"] = data["current_value"].astype(float)
        return (
            data.groupby("episode_id")["current_value"]
            .last()
            .loc[data["episode_id"]]
            .values.reshape(-1)
        )

    def calculate_tail_values(
        self,
        data_buffer: DataBuffer,
    ) -> np.array:
        data = data_buffer.to_pandas(keys=["episode_id", "current_value"])
        data["episode_id"] = data["episode_id"].astype(int)
        data["current_value"] = data["current_value"].astype(float)

        groupby_episode_values = data.groupby(["episode_id"])["current_value"]

        last_values = (
            groupby_episode_values.last().loc[data["episode_id"]].values.reshape(-1)
        )
        current_values_shifted = groupby_episode_values.shift(
            periods=1, fill_value=0.0
        ).values.reshape(-1)

        return last_values - current_values_shifted

    def calculate_baseline(self, data_buffer: DataBuffer) -> np.array:
        r"""Computes the baseline value used for scaling the advantages during optimization.

        Args:
            data_buffer: The data buffer containing trajectory information.

        Returns:
            Array of baselines $b_{[0 : T]}$, where
                $b_t = \frac{1}{N} \sum_{j = 1} ^ {M}  \sum_{t' = t} ^ T \gamma ^ {t \delta t} r(\obs^j_t, \action^j_t)$.
                $\delta t$ is sampling time, $\obs^j_t$ and $\action^j_t$ are the observation and the action
                from step t and episode $j$ from the **previous iteration**, $\gamma$ is discount factor,
                $M$ is the number of episodes.

        Notes:
            The initial value of baseline is set to zero,
        """

        baseline = self.next_baseline
        if not self.is_do_not_let_the_past_distract_you:
            self.next_baseline = np.mean(data_buffer.to_pandas(keys=["value"]).values)

        else:
            self.next_baseline = (
                data_buffer.to_pandas(keys=["tail_value", "step_id"])
                .astype(float)
                .groupby("step_id")
                .mean()
                .loc[
                    data_buffer.to_pandas(keys=["step_id"])
                    .astype(float)
                    .values.reshape(-1)
                ]
            ).values
        if isinstance(baseline, float):
            return np.full(shape=len(data_buffer), fill_value=baseline)
        else:
            return baseline

    def data_buffer_objective_keys(self) -> List[str]:
        """Define the specific keys required from the data buffer to calculate the objective.

        Returns:
            A list of string keys corresponding to the required data: observation, action, tail_value, value, and baseline.
        """
        return [
            "observation",
            "action",
            "tail_value",
            "value",
            "baseline",
        ]

    def objective_function(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        tail_value: torch.Tensor,
        value: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the surrogate loss using the reinforcement learning objective, as defined by the REINFORCE algorithm.

        Args:
            observation: The observations received from the environment.
            action: The actions taken in the environment.
            tail_value: The tail values calculated for the trajectories.
            value: The values of the current state-action pairs.
            baseline: The baseline values used for scaling.

        Returns:
            A loss value that, when minimized, leads to policy improvement.

        Note:
            All the arguments are sampled from [`DataBuffer`][regelum.data_buffers.DataBuffer]
            via [`iter_bathes`][regelum.data_buffers.DataBuffer.iter_batches] method
        """
        return reinforce_objective(
            policy_model=self.model,
            observations=observation,
            actions=action,
            tail_values=tail_value,
            values=value,
            baselines=baseline,
            is_with_baseline=self.is_with_baseline,
            is_do_not_let_the_past_distract_you=self.is_do_not_let_the_past_distract_you,
            N_episodes=self.N_episodes,
        )


class PolicySDPG(PolicyGradient):
    """Policy for Stochastic Deep Policy Gradient (SDPG)

    Extends PolicyGradient via using a critic for temporal difference objective."""

    def __init__(
        self,
        model: ModelNN,
        critic: Critic,
        system: Union[System, ComposedSystem],
        optimizer_config: OptimizerConfig,
        sampling_time: float,
        discount_factor: float = 1.0,
        device: str = "cpu",
        is_normalize_advantages: bool = True,
        gae_lambda: float = 1.0,
    ):
        """Instantiate SDPG class.

        Args:
            model: Policy model used to compute actions.
            critic: Critic object used in the temporal difference objective.
            system: The environment or system where the policy is deployed.
            optimizer_config: Settings for the associated optimization procedure.
            sampling_time: Time interval used to discretize continuous time into steps.
            discount_factor: Discount factor used to compute the present value of future running objectives.
            device: The computation device ('cpu' or 'cuda') to execute the optimization process.
            is_normalize_advantages: Whether to normalize the advantages.
            gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            discount_factor=discount_factor,
            device=device,
            critic=critic,
            optimizer_config=optimizer_config,
        )
        self.sampling_time = sampling_time
        self.is_normalize_advantages = is_normalize_advantages
        self.gae_lambda = gae_lambda
        self.initialize_optimization_procedure()

    def data_buffer_objective_keys(self) -> List[str]:
        """Specifies the keys in the data buffer required for computing the optimization objective.

        Returns:
            A list of strings that are keys in the data buffer for retrieving necessary data.

        Note:
            The specified keys are 'observation', 'action', 'time', 'episode_id', and 'running_objective'.
        """
        return ["observation", "action", "time", "episode_id", "running_objective"]

    def objective_function(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        time: torch.Tensor,
        episode_id: torch.Tensor,
        running_objective: torch.Tensor,
    ) -> torch.Tensor:
        """Defines the objective function to optimize policy under the SDPG framework using the provided data buffer components.

        Args:
            observation: The most recent observations received from the environment.
            action: The actions executed in the environment based on the policy.
            time: The timestep or moment when the action is taken.
            episode_id: Unique identifiers for episodes in the training data.
            running_objective: Cumulative reward or running objective value of the taken actions.

        Returns:
            The calculated objective value from the SDPG algorithm for optimization purposes.

        Notes:
            The objective function gets arguments from data buffer.
        """

        return sdpg_objective(
            policy_model=self.model,
            critic_model=self.critic.model,
            observations=observation,
            actions=action,
            times=time,
            discount_factor=self.discount_factor,
            N_episodes=self.N_episodes,
            episode_ids=episode_id.long(),
            running_objectives=running_objective,
            sampling_time=self.sampling_time,
            is_normalize_advantages=self.is_normalize_advantages,
            gae_lambda=self.gae_lambda,
        )


class PolicyPPO(PolicyGradient):
    """Implements the Proximal Policy Optimization (PPO) algorithm"""

    def __init__(
        self,
        model: ModelNN,
        critic: Critic,
        system: Union[System, ComposedSystem],
        optimizer_config: OptimizerConfig,
        sampling_time: float,
        discount_factor: float = 1.0,
        device: str = "cpu",
        gae_lambda: float = 0.0,
        running_objective_type: str = "cost",
        cliprange: float = 0.2,
        is_normalize_advantages: bool = True,
        entropy_coeff: float = 0.0,
    ):
        """Instantiate PPO policy class.

        Args:
            model: An instance of the policy model.
            critic: An instance of the critic used for optimization.
            system: The environment or system the policy will interact with.
            optimizer_config: Configuration parameters for the optimizer.
            sampling_time:  Sampling period used for discretizing continuous domains.
            discount_factor: Factor to discount future rewards.
            device: The device to perform computation on (CPU or GPU).
            gae_lambda: Lambda parameter for Generalized Advantage Estimation (GAE).
            running_objective_type: Type of the running objective, can be 'cost' or 'reward'.
            cliprange: Range used for clipping in PPO algorithm.
            is_normalize_advantages: Whether to normalize the advantages.
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            discount_factor=discount_factor,
            device=device,
            critic=critic,
            optimizer_config=optimizer_config,
        )
        self.sampling_time = sampling_time
        self.cliprange = cliprange
        self.running_objective_type = running_objective_type
        self.gae_lambda = gae_lambda
        self.is_normalize_advantages = is_normalize_advantages
        self.entropy_coeff = entropy_coeff
        self.initialize_optimization_procedure()

    def data_buffer_objective_keys(self) -> List[str]:
        """Return a list of data keys that will be used in the objective function of PPO optimization.

        Returns:
            A list of string keys for retrieving necessary data from the buffer for optimization.
        """

        return [
            "observation",
            "action",
            "time",
            "episode_id",
            "running_objective",
            "initial_log_probs",
        ]

    def objective_function(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        time: torch.Tensor,
        episode_id: torch.Tensor,
        running_objective: torch.Tensor,
        initial_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Define the surrogate objective function used for PPO optimization.

        Args:
            observation: Observations from the environment.
            action: Actions taken by the policy.
            time: Time steps of the action.
            episode_id: Identifiers for episodes within the buffer.
            running_objective: Cumulative reward or cost for each step in the episodes.
            initial_log_probs: Initial policy log probabilities of actions.

        Returns:
            The surrogate objective function value for the PPO algorithm.
        """
        return ppo_objective(
            policy_model=self.model,
            critic_model=self.critic.model,
            observations=observation,
            actions=action,
            times=time,
            discount_factor=self.discount_factor,
            N_episodes=self.N_episodes,
            episode_ids=episode_id.long(),
            running_objectives=running_objective,
            initial_log_probs=initial_log_probs,
            cliprange=self.cliprange,
            running_objective_type=self.running_objective_type,
            sampling_time=self.sampling_time,
            gae_lambda=self.gae_lambda,
            is_normalize_advantages=self.is_normalize_advantages,
            entropy_coeff=self.entropy_coeff,
        )

    def update_data_buffer(self, data_buffer: DataBuffer) -> None:
        """Perform updates to the data buffer, preparing the necessary data for PPO optimization, such as log probabilities.

        Args:
            data_buffer: The buffer containing interaction data between the agent and the environment.

        """
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


class PolicyDDPG(PolicyGradient):
    """Policy for Deterministic Deep Policy Gradient (DDPG)."""

    def __init__(
        self,
        model: ModelNN,
        critic: Critic,
        system: Union[System, ComposedSystem],
        optimizer_config: OptimizerConfig,
        discount_factor: float = 1.0,
        device: str = "cpu",
    ):
        """Instantiate DDPG class.

        Args:
            model: The neural network policy model.
            critic: The critic network used to evaluate the action value function.
            system: The environment where the policy is deployed.
            optimizer_config: Configuration for the optimization process.
            discount_factor: The discount factor for future rewards, generally between 0 and 1.
            device: The computation device, typically 'cpu' or 'cuda'.
        """
        PolicyGradient.__init__(
            self,
            model=model,
            system=system,
            discount_factor=discount_factor,
            device=device,
            critic=critic,
            optimizer_config=optimizer_config,
        )
        self.initialize_optimization_procedure()

    def data_buffer_objective_keys(self) -> List[str]:
        """Specify the keys from the data buffer that are required for computing the objective function for DDPG optimization.

        Returns:
            A list of string keys representing the required data from the data buffer.
        """
        return ["observation"]

    def objective_function(self, observation: torch.Tensor) -> torch.Tensor:
        """Calculate the objective function for DDPG, using the provided observations data to optimize the policy.

        Args:
            observation: The observations received from the environment.

        Returns:
            The objective value based on the DDPG algorithm for policy optimization.
        """

        return ddpg_objective(
            policy_model=self.model,
            critic_model=self.critic.model,
            observations=observation,
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
        epsilon_random_parameter: Optional[float] = None,
        algorithm: str = "mpc",
    ):
        """Initializes the RLPolicy.

        Args:
            model: The model representing the predictive policy.
            critic: The critic used for reinforcement learning, which can be value-based or action-based.
            system: The environment where the policy operates.
            action_bounds: Limits on the range of possible actions.
            optimizer_config: Configuration settings for the optimization procedure.
            predictor: Optional predictor used to forecast state sequences.
            prediction_horizon: Duration over which predictions are made.
            running_objective: The objective function used in control problems.
            constraint_parser: Optional parser for parsing and handling constraints.
            discount_factor: Discount factor for future rewards in control problems.
            device: The computational device to be used, either "cpu" or "cuda".
            epsilon_random_parameter: The probability parameter for the epsilon-greedy policy.
            algorithm: The type of RL algorithm to use, such as "mpc", "rpv", "sql", "rql", "greedy".
        """

        Policy.__init__(
            self,
            model=model,
            system=system,
            action_bounds=action_bounds,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
        )

        self.action_bounds = np.array(action_bounds)
        self.predictor = predictor
        self.prediction_horizon = prediction_horizon
        self.critic = critic
        self.device = device
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
        """Keys to sample from data buffer for optimization.

        Identifies the required keys in the data buffer for calculating the optimization objective under the
        reinforcement learning policy.

        Returns:
            A list of string keys required from the data buffer for optimization.
        """
        return ["observation", "estimated_state"]

    def initialize_optimization_procedure(self):
        """Prepare the optimization environment by setting up the objective function and constraints.

        Note:
            Initializes problem-specific variables required for the optimization process, which will vary based on the algorithm used.
        """

        objective_variables = []
        self.observation_variable = self.create_variable(
            1, self.system.dim_observation, name="observation", is_constant=True
        )
        self.estimated_state_variable = self.create_variable(
            1, self.system.dim_state, name="estimated_state", is_constant=True
        )
        objective_variables.extend(
            [self.observation_variable, self.estimated_state_variable]
        )
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
            if self.algorithm == "rpv":
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
                state=self.estimated_state_variable,
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
        self,
        estimated_state: RgArray,
        observation: RgArray,
        policy_model_weights: Weights,
        critic_weights: Optional[Weights] = None,
    ) -> RgArray:
        """
        Provides the interface for calculating the RL-based objective function for policy optimization.

        Args:
            estimated_state: The estimated current state of the system.
            observation: Latest observed state from the environment.
            policy_model_weights: Weights of the policy model, typically obtained from an optimizer.
            critic_weights: Weights of the critic model, used if the algorithm is critic-based.

        Returns:
            The calculated objective based on the specified RL algorithm.
        """
        if self.algorithm == "mpc":
            return mpc_objective(
                estimated_state=estimated_state,
                observation=observation,
                policy_model_weights=policy_model_weights,
                predictor=self.predictor,
                running_objective=self.running_objective,
                model=self.model,
                prediction_horizon=self.prediction_horizon,
                discount_factor=self.discount_factor,
            )
        elif self.algorithm == "rpv":
            return rpv_objective(
                estimated_state=estimated_state,
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
                estimated_state=estimated_state,
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
                estimated_state=estimated_state,
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
            return rg.mean(
                self.critic(
                    observation,
                    self.model(observation, weights=policy_model_weights),
                    weights=critic_weights,
                )
            )
        else:
            raise AssertionError("RLPolicy: Wrong algorithm name")

    def optimize(self, data_buffer: DataBuffer) -> None:
        """Run the optimization procedure to update the policy model using the specified RL algorithm and data collected in the buffer.

        Args:
            data_buffer: The data buffer containing trajectories collected during policy execution.
        """
        opt_kwargs = data_buffer.get_optimization_kwargs(
            keys=self.data_buffer_objective_keys(),
            optimizer_config=self.optimizer_config,
        )

        if opt_kwargs is not None:
            if self.kind == "symbolic":
                result = (
                    super().optimize(
                        **opt_kwargs,
                        policy_model_weights=self.get_initial_guess(),
                        tol=1e-8,
                    )
                    if self.critic.weights is None
                    else super().optimize(
                        **opt_kwargs,
                        policy_model_weights=self.get_initial_guess(),
                        critic_weights=self.critic.weights,
                        tol=1e-8,
                    )
                )
                self.update_weights(result["policy_model_weights"])
            elif self.kind == "tensor":
                super().optimize(**opt_kwargs)

    def get_initial_guess(self):
        """Generates an initial guess for the policy parameters to start the optimization process.

        Returns:
            A numpy array or similar data structure representing the initial guess for policy weights.
        """

        return self.model.weights


class KinPointStabilizingPolicy(Policy):
    r"""Scenario for kinematic point stabilization.

    Implements the following policy: $\action = -\text{gain} \cdot \observation$
    """

    def __init__(self, gain):
        """Initialize an instance of the class with the given gain.

        Args:
            gain: The gain value to set for the instance.

        Returns:
            None
        """
        super().__init__()
        self.gain = gain

    def get_action(self, observation):
        return -self.gain * observation


class ThreeWheeledWRobotKinematicStabilizingPolicy(Policy):
    """Scenario for non-inertial three-wheeled robot composed of three PID scenarios."""

    def __init__(self, K):
        """Initialize an instance of scenario.

        Args:
            K: gain of scenario
        """
        super().__init__()
        self.K = K

    def get_action(self, observation):
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
                * rg.sqrt(rg.abs(angle - angle_cond))
            )
            v = 0
        elif not np.allclose((x, y), (0, 0), atol=1e-03) and np.isclose(
            angle, angle_cond, atol=1e-03
        ):
            omega = 0
            v = -self.K * rg.sqrt(rg.norm_2(rg.hstack([x, y])))
        elif np.allclose((x, y), (0, 0), atol=1e-03) and not np.isclose(
            angle, 0, atol=1e-03
        ):
            omega = -self.K * np.sign(angle) * rg.sqrt(rg.abs(angle))
            v = 0
        else:
            omega = 0
            v = 0

        return rg.force_row(rg.hstack([v, omega]))


class InvertedPendulumStabilizingPolicy(Policy):
    """A nominal policy for inverted pendulum representing a PD controller."""

    def __init__(self, gain):
        """Initialize an instance of policy.

        Args:
            gain: gain of PID controller.
        """
        super().__init__()
        self.gain = gain

    def get_action(self, observation: np.array):
        return np.array(
            [[-((observation[0, 0]) + 0.1 * (observation[0, 1])) * self.gain]]
        )


class ThreeWheeledWRobotNIDisassembledCLFPolicy(Policy):
    """Nominal parking scenario for NI using disassembled control Lyapunov function."""

    def __init__(self, scenario_gain=10):
        """Initialize an instance of disassembled-clf scenario.

        Args:
            scenario_gain: gain of scenario
        """
        super().__init__()
        self.scenario_gain = scenario_gain

    def _zeta(self, xNI: np.array) -> np.array:
        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(abs(xNI[2]))

        nablaL = rg.zeros(3)

        nablaL[0] = (
            4 * xNI[0] ** 3
            + rg.abs(xNI[2]) ** 3
            / sigma**3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[0]
        )
        nablaL[1] = (
            4 * xNI[1] ** 3
            + rg.abs(xNI[2]) ** 3
            / sigma**3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[1]
        )
        nablaL[2] = 3 * rg.abs(xNI[2]) ** 2 * rg.sign(xNI[2]) + rg.abs(
            xNI[2]
        ) ** 3 / sigma**3 * 1 / np.sqrt(rg.abs(xNI[2])) * rg.sign(xNI[2])

        theta = 0

        sigma_tilde = (
            xNI[0] * rg.cos(theta) + xNI[1] * rg.sin(theta) + np.sqrt(rg.abs(xNI[2]))
        )

        nablaF = rg.zeros(3)

        nablaF[0] = (
            4 * xNI[0] ** 3 - 2 * rg.abs(xNI[2]) ** 3 * rg.cos(theta) / sigma_tilde**3
        )
        nablaF[1] = (
            4 * xNI[1] ** 3 - 2 * rg.abs(xNI[2]) ** 3 * rg.sin(theta) / sigma_tilde**3
        )
        nablaF[2] = (
            (
                3 * xNI[0] * rg.cos(theta)
                + 3 * xNI[1] * rg.sin(theta)
                + 2 * np.sqrt(rg.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * rg.sign(xNI[2])
            / sigma_tilde**3
        )

        if xNI[0] == 0 and xNI[1] == 0:
            return nablaF
        else:
            return nablaL

    def _kappa(self, xNI: np.array) -> np.array:
        kappa_val = rg.zeros(2)

        G = rg.zeros([3, 2])
        G[:, 0] = rg.array([1, 0, xNI[1]], prototype=G)
        G[:, 1] = rg.array([0, 1, -xNI[0]], prototype=G)

        zeta_val = self._zeta(xNI)

        kappa_val[0] = -rg.abs(np.dot(zeta_val, G[:, 0])) ** (1 / 3) * rg.sign(
            rg.dot(zeta_val, G[:, 0])
        )
        kappa_val[1] = -rg.abs(np.dot(zeta_val, G[:, 1])) ** (1 / 3) * rg.sign(
            rg.dot(zeta_val, G[:, 1])
        )

        return kappa_val

    def _F(self, xNI, eta, theta):
        sigma_tilde = (
            xNI[0] * rg.cos(theta) + xNI[1] * rg.sin(theta) + np.sqrt(rg.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + rg.abs(xNI[2]) ** 3 / sigma_tilde**2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * rg.dot(z, z)

    def _Cart2NH(self, coords_Cart):
        xNI = rg.zeros(3)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        angle = coords_Cart[2]

        xNI[0] = angle
        xNI[1] = xc * rg.cos(angle) + yc * rg.sin(angle)
        xNI[2] = -2 * (yc * rg.cos(angle) - xc * rg.sin(angle)) - angle * (
            xc * rg.cos(angle) + yc * rg.sin(angle)
        )

        return xNI

    def _NH2ctrl_Cart(self, xNI, uNI):
        uCart = rg.zeros(2)

        uCart[0] = uNI[1] + 1 / 2 * uNI[0] * (xNI[2] + xNI[0] * xNI[1])
        uCart[1] = uNI[0]

        return uCart

    def get_action(self, observation):
        xNI = self._Cart2NH(observation[0])
        kappa_val = self._kappa(xNI)
        uNI = self.scenario_gain * kappa_val

        return self._NH2ctrl_Cart(xNI, uNI).reshape(1, -1)

    def compute_LF(self, observation):
        xNI = self._Cart2NH(observation)

        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(rg.abs(xNI[2]))
        LF_value = xNI[0] ** 4 + xNI[1] ** 4 + rg.abs(xNI[2]) ** 3 / sigma**2

        return LF_value


class MemoryPIDPolicy(Policy):
    """A base class for PID scenario.

    This scenario is able to use stored data in order to detect whether system is stabilized or not.
    """

    def __init__(
        self,
        P: float,
        I: float,
        D: float,
        setpoint: Union[np.array, float] = 0.0,
        sampling_time: float = 0.01,
        initial_point: Union[np.array, List[float]] = (-5, -5),
        buffer_length: int = 30,
    ):
        """Initialize an instance of ScenarioMemoryPID.

        Whatever

        Args:
            P: proportional gain
            I: integral gain
            D: differential gain
            setpoint: point using as target turing error evaluation
            sampling_time: time interval between two consecutive actions
            initial_point: point at which computations has begun
            buffer_length: length of stored buffer
        """
        super().__init__()
        self.P = P
        self.I = I
        self.D = D

        self.setpoint = setpoint
        self.integral = 0.0
        self.error_old = 0.0
        self.sampling_time = sampling_time
        self.initial_point = initial_point
        if isinstance(initial_point, (float, int)):
            self.observation_size = 1
        else:
            self.observation_size = len(initial_point)

        self.buffer_length = buffer_length
        self.observation_buffer = rg.ones((self.observation_size, buffer_length)) * 1e3

    def compute_error(self, process_variable):
        if isinstance(process_variable, (float, int)):
            error = process_variable - self.setpoint
        else:
            if len(process_variable) == 1:
                error = process_variable - self.setpoint
            else:
                norm = rg.norm_2(self.setpoint - process_variable)
                error = norm * rg.sign(rg.dot(self.initial_point, process_variable))
        return error

    def compute_integral(self, error):
        self.integral += error * self.sampling_time
        return self.integral

    def compute_error_derivative_numerically(self, error):
        error_derivative = (error - self.error_old) / self.sampling_time
        self.error_old = error
        return error_derivative

    def error_derivative(self, error):
        return None

    def compute_signal(
        self,
        process_variable,
        error_derivative=None,
    ):
        error = self.compute_error(process_variable)
        integral = self.compute_integral(error)

        if error_derivative is None:
            error_derivative = self.compute_error_derivative_numerically(error)

        PID_signal = -(self.P * error + self.I * integral + self.D * error_derivative)

        return PID_signal

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def update_observation_buffer(self, observation):
        self.observation_buffer = rg.push_vec(self.observation_buffer, observation)

    def set_initial_point(self, point):
        self.initial_point = point

    def reset(self):
        self.integral = 0.0
        self.error_old = 0.0

    def reset_buffer(self):
        self.observation_buffer = (
            rg.ones((self.observation_size, self.buffer_length)) * 1e3
        )

    def is_stabilized(self, stabilization_tollerance=1e-3):
        is_stabilized = np.allclose(
            self.observation_buffer,
            rg.rep_mat(rg.reshape(self.setpoint, (-1, 1)), 1, self.buffer_length),
            atol=stabilization_tollerance,
        )
        return is_stabilized


class ThreeWheeledRobotPIDPolicy:
    """Nominal stabilzing policy for ThreeWheeledRobotDynamic inertial system."""

    def __init__(
        self,
        state_init: np.array,
        PID_arctg_params: Tuple[float, float, float] = (10, 0.0, 3),
        PID_v_zero_params: Tuple[float, float, float] = (35, 0.0, 1.2),
        PID_x_y_origin_params: Tuple[float, float, float] = (35, 0.0, 35),
        PID_angle_origin_params: Tuple[float, float, float] = (30, 0.0, 10),
        v_to_zero_bounds: Tuple[float, float, float] = (0.0, 0.05),
        to_origin_bounds: Tuple[float, float, float] = (0.0, 0.1),
        to_arctan_bounds: Tuple[float, float, float] = (0.01, 0.2),
    ):
        """Initialize object.

        Args:
            state_init: initial state of 3wrobot_dyn
            PID_arctg_params: coefficients for PD scenario which sets
                the direction of robot to origin
            PID_v_zero_params: coefficients for PD scenario which forces
                speed to zero as robot moves to origin
            PID_x_y_origin_params: coefficients for PD scenario which
                moves robot to origin
            PID_angle_origin_params: coefficients for PD scenario which
                sets angle to zero near origin
            v_to_zero_bounds: bounds for enabling scenario which
                decelerates
            to_origin_bounds: bounds for enabling scenario which moves
                robot to origin
            to_arctan_bounds: bounds for enabling scenario which direct
                robot to origin
        """

        self.v_to_zero_bounds = v_to_zero_bounds
        self.to_origin_bounds = to_origin_bounds
        self.to_arctan_bounds = to_arctan_bounds

        self.state_init = state_init

        self.action_old = rg.zeros(2)
        self.PID_angle_arctan = MemoryPIDPolicy(
            *PID_arctg_params, initial_point=self.state_init[2]
        )
        self.PID_v_zero = MemoryPIDPolicy(
            *PID_v_zero_params, initial_point=self.state_init[3], setpoint=0.0
        )
        self.PID_x_y_origin = MemoryPIDPolicy(
            *PID_x_y_origin_params,
            setpoint=rg.array([0.0, 0.0]),
            initial_point=self.state_init[:2],
        )
        self.PID_angle_origin = MemoryPIDPolicy(
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
            return rg.array([0.0])
        F_error_derivative = (
            v * (x * rg.cos(angle) + y * rg.sin(angle)) / rg.sqrt(x**2 + y**2)
        ) * rg.sign(rg.dot(self.PID_x_y_origin.initial_point, [x, y]))
        return rg.array([F_error_derivative])

    def get_action(self, observation):
        observation = observation[0]
        x = observation[0]
        y = observation[1]
        angle = rg.array([observation[2]])
        v = rg.array([observation[3]])
        omega = rg.array([observation[4]])
        (F_min, F_max), (M_min, M_max) = self.action_bounds[0], self.action_bounds[1]

        self.PID_x_y_origin.set_initial_point(rg.array([x, y]))
        F_error_derivative = self.F_error_derivative(x, y, angle[0], v[0])
        M_arctan_error_derivative = omega
        F_v_to_zero_error_derivative = self.current_F / self.m

        F = self.PID_x_y_origin.compute_signal(
            [x, y], error_derivative=F_error_derivative
        )
        angle_setpoint = rg.array([self.get_setpoint_for_PID_angle_arctan(x, y)])
        self.PID_angle_arctan.set_setpoint(angle_setpoint)
        M_arctan = self.PID_angle_arctan.compute_signal(
            angle, error_derivative=M_arctan_error_derivative
        )
        M_zero = self.PID_angle_origin.compute_signal(angle, error_derivative=omega[0])
        F_v_to_zero = self.PID_v_zero.compute_signal(
            v, error_derivative=F_v_to_zero_error_derivative
        )

        lbd_v_to_zero = self.cdf_uniform(
            rg.norm_2(rg.array([x, y, v[0]])),
            self.v_to_zero_bounds[0],
            self.v_to_zero_bounds[1],
        )
        lbd = self.cdf_uniform(
            rg.norm_2(rg.array([x, y])),
            self.to_origin_bounds[0],
            self.to_origin_bounds[1],
        )
        lbd_arctan = self.cdf_uniform(
            rg.abs(angle[0] - self.PID_angle_arctan.setpoint[0]),
            self.to_arctan_bounds[0],
            self.to_arctan_bounds[1],
        )
        control_to_origin = rg.array(
            [
                (1 - lbd_arctan) * np.clip(F[0], F_min, F_max)
                + lbd_arctan * np.clip(F_v_to_zero[0], F_min, F_max),
                lbd_arctan * np.clip(M_arctan[0], M_min, M_max),
            ]
        )
        control_v_to_zero = rg.array([np.clip(F_v_to_zero[0], F_min, F_max), 0.0])
        control_angle_to_zero = rg.array([0, np.clip(M_zero[0], M_min, M_max)])

        action = (
            lbd * control_to_origin
            + (1 - lbd) * control_v_to_zero
            + control_angle_to_zero * (1 - lbd_v_to_zero)
        )

        self.current_F = action[0]
        self.current_M = action[1]

        return action.reshape(1, -1)


class CartPoleEnergyBasedPolicy(Policy):
    """An energy-based scenario for cartpole."""

    def __init__(
        self,
        scenario_gain: float = 10,
        upright_gain=None,
        swingup_gain=10,
        pid_loc_thr=0.35,
        pid_scale_thr=10.0,
        clip_bounds=(-1, 1),
    ):
        """Initialize an instance of ScenarioCartPoleEnergyBased.

        Args:
            action_bounds: upper and lower bounds for action yielded
                from policy
            sampling_time: time interval between two consecutive actions
            scenario_gain: scenario gain
            system: an instance of Cartpole system
        """
        super().__init__()
        from regelum.system import CartPole

        self.scenario_gain = scenario_gain
        self.m_c, self.m_p, self.g, self.l = (
            CartPole().parameters["m_c"],
            CartPole().parameters["m_p"],
            CartPole().parameters["g"],
            CartPole().parameters["l"],
        )
        self.upright_gain = upright_gain
        self.swingup_gain = swingup_gain
        self.pid_loc_thr = pid_loc_thr
        self.pid_scale_thr = pid_scale_thr
        self.clip_bounds = clip_bounds

    def get_action(self, observation):
        observation = observation[0]

        # sin_theta, one_minus_cos_theta, x, theta_dot, x_dot = observation
        sin_theta, one_minus_cos_theta, theta_dot, x_dot = observation
        x = 0
        cos_theta = 1 - one_minus_cos_theta
        theta = rg.atan2(sin_theta, cos_theta)
        # E_total = (
        #     self.m_p * self.l**2 * theta_dot**2 / 2
        #     + self.m_p * self.g * self.l * (1 - cos_theta)
        # )
        # print(E_total)
        lbd = (1 - rg.tanh((theta - self.pid_loc_thr) * self.pid_scale_thr)) / 2

        low, high = self.clip_bounds
        x_clipped = rg.clip(x, low, high)
        x_dot_clipped = rg.clip(x_dot, low, high)
        # self.action = (1 - lbd) * (
        #     self.swingup_gain * E_total * rg.sign(cos_theta * theta_dot)
        # ) + lbd * self.upright_gain.T @ rg.array(
        #     [theta, x_clipped, theta_dot, x_dot_clipped]
        # )

        # self.action = self.upright_gain.T @ rg.array(
        #     [theta, x_clipped, theta_dot, x_dot_clipped]
        # )
        self.upswing_gain = 3
        if cos_theta < 0:
            action_upswing = rg.sign(theta_dot) * self.upswing_gain
        else:
            action_upswing = rg.sign(sin_theta) * self.upswing_gain

        # if (cos_theta > 0 and theta_dot > 0) or (cos_theta > 0 and theta_dot < 0):
        #     action_upswing = -1 * action_upswing

        # self.upright_gain = np.array([50, 2., 20., 3.])
        # self.upright_gain = np.array([500, 0.0, 200.0, 0.0])

        action_upright = self.upright_gain.T @ rg.array(
            [theta, x_clipped, theta_dot, x_dot_clipped]
        )

        self.action = (1 - lbd) * action_upswing + lbd * action_upright

        return self.action.reshape(1, -1)


class ConstantPolicy(Policy):
    def __init__(self, value):
        super().__init__()
        self.value = np.array(value).reshape(1, -1)

    def get_action(self, observation):
        return self.value


class LunarLanderPIDPolicy(Policy):
    """Nominal PID scenario for lunar lander."""

    def __init__(
        self,
        state_init,
        PID_angle_parameters=None,
        PID_height_parameters=None,
        PID_x_parameters=None,
    ):
        """Initialize an instance of PID scenario for lunar lander.

        Args:
            action_bounds: upper and lower bounds for action yielded
                from policy
            state_init: state at which simulation has begun
            sampling_time: time interval between two consecutive actions
            PID_angle_parameters: parameters for PID scenario
                stabilizing angle of lander
            PID_height_parameters: parameters for PID scenario
                stabilizing y-coordinate of lander
            PID_x_parameters: parameters for PID scenario stabilizing
                x-coordinate of lander
        """
        super().__init__()

        if PID_angle_parameters is None:
            PID_angle_parameters = [1, 0, 0]
        if PID_height_parameters is None:
            PID_height_parameters = [10, 0, 0]
        if PID_x_parameters is None:
            PID_x_parameters = [10, 0, 0]
        self.PID_angle = MemoryPIDPolicy(
            *PID_angle_parameters,
            initial_point=rg.array([state_init[2]]),
            setpoint=rg.array([0]),
        )
        self.PID_height = MemoryPIDPolicy(
            *PID_height_parameters,
            initial_point=rg.array([state_init[1]]),
            setpoint=rg.array([0]),
        )
        self.PID_x = MemoryPIDPolicy(
            *PID_x_parameters,
            initial_point=rg.array([state_init[2]]),
            setpoint=rg.array([0]),
        )

    def get_action(self, observation):
        observation = observation[0]
        self.action = [0, 0]

        if abs(observation[2]) > self.threshold:
            self.threshold = self.threshold_1
            self.action[0] = self.PID_angle.compute_signal(
                rg.array([observation[2]]), error_derivative=observation[5]
            )[0]

        else:
            self.threshold = self.threshold_2
            self.action[0] = self.PID_x.compute_signal(
                rg.array([observation[0]]), error_derivative=observation[3]
            )[0]
            self.action[1] = self.PID_height.compute_signal(
                rg.array([observation[1]]), error_derivative=observation[4]
            )[0]

        self.action = rg.array(self.action).reshape(1, -1)

        return self.action


class LunarLanderPIDPolicyNew(Policy):
    """Nominal PID scenario for lunar lander."""

    def __init__(
        self,
        state_init,
        PID_angle_parameters=None,
        PID_omega_parameters=None,
        PID_height_parameters=None,
        PID_y_dot_parameters=None,
        PID_x_parameters=None,
    ):
        """Initialize an instance of PID scenario for lunar lander.

        Args:
            action_bounds: upper and lower bounds for action yielded
                from policy
            state_init: state at which simulation has begun
            sampling_time: time interval between two consecutive actions
            PID_angle_parameters: parameters for PID scenario
                stabilizing angle of lander
            PID_height_parameters: parameters for PID scenario
                stabilizing y-coordinate of lander
            PID_x_parameters: parameters for PID scenario stabilizing
                x-coordinate of lander
        """
        super().__init__()
        self.PID_omega = MemoryPIDPolicy(
            *PID_omega_parameters,
            initial_point=rg.array([state_init[5]]),
            setpoint=rg.array([0]),
        )
        self.PID_angle = MemoryPIDPolicy(
            *PID_angle_parameters,
            initial_point=rg.array([state_init[5]]),
            setpoint=rg.array([0]),
        )
        self.PID_y_dot = MemoryPIDPolicy(
            *PID_y_dot_parameters,
            initial_point=rg.array([state_init[4]]),
            setpoint=rg.array([0]),
        )
        self.PID_height = MemoryPIDPolicy(
            *PID_height_parameters,
            initial_point=rg.array([state_init[1]]),
            setpoint=rg.array([5]),
        )
        self.PID_x = MemoryPIDPolicy(
            *PID_x_parameters,
            initial_point=rg.array([state_init[0]]),
            setpoint=rg.array([0]),
        )
        self.action = rg.array([0.0, 0.0]).reshape(1, 2)

    def get_action(self, observation):
        new_support_lvl = rg.min(
            rg.array(
                [
                    10
                    * rg.sqrt(
                        rg.norm_2(
                            rg.hstack((observation[0, 2:5], observation[0, 0])) ** 2
                        )
                    ),
                    3.0,
                ]
            )
        )
        self.PID_height.set_setpoint(new_support_lvl)

        self.action[0, 0] = self.PID_omega.compute_signal(
            rg.array([observation[0, 2]]),
            error_derivative=rg.array([observation[0, 5]]),
        ).reshape(-1) - rg.cos(observation[0, 2]) ** 2 * self.PID_x.compute_signal(
            rg.array([observation[0, 0]]),
            error_derivative=rg.array([observation[0, 3]]),
        ).reshape(
            -1
        )
        self.action[0, 1] = rg.sign(
            rg.cos(observation[0, 2])
        ) * self.PID_y_dot.compute_signal(rg.array([observation[0, 4]])).reshape(
            -1
        ) + rg.sign(
            rg.cos(observation[0, 2])
        ) * self.PID_height.compute_signal(
            rg.array([observation[0, 1]]),
            error_derivative=rg.array([observation[0, 4]]),
        ).reshape(
            -1
        )

        return self.action


class TwoTankPIDPolicy(Policy):
    """PID scenario for double tank system."""

    def __init__(
        self,
        state_init: np.array = None,
        PID_2tank_parameters_x1=(1, 0, 0),
        PID_2tank_parameters_x2=(1, 0, 0),
    ):
        """Initialize an instance of Scenario2TankPID.

        Args:
            action_bounds: upper and lower bounds for action yielded
                from policy
            params: parameters of double tank system
            state_init: state at which simulation has begun
            sampling_time: time interval between two consecutive actions
            PID_2tank_parameters_x1: parameters for PID scenario
                stabilizing first component of system's state
            PID_2tank_parameters_x2: parameters for PID scenario
                stabilizing second component of system's state
        """
        from regelum.system import TwoTank

        super().__init__()
        params = TwoTank._parameters

        self.K1 = params["K1"]
        self.K2 = params["K2"]
        self.K3 = params["K3"]
        self.tau1 = params["tau1"]
        self.tau2 = params["tau2"]

        self.state_init = state_init
        self.PID_2tank_x1 = MemoryPIDPolicy(
            *PID_2tank_parameters_x1,
            initial_point=rg.array([state_init[0]]),
        )
        self.PID_2tank_x2 = MemoryPIDPolicy(
            *PID_2tank_parameters_x2,
            initial_point=rg.array([state_init[1]]),
        )

        self.action = np.zeros((1, 1))

    def get_action(self, observation):
        error_derivative_x1 = -(
            1 / (self.tau1) * (-observation[0][0] + self.K1 * self.action[0][0])
        )
        error_derivative_x2 = (
            -1
            / (self.tau2)
            * (
                -observation[0][1]
                + self.K2 * observation[0][0]
                + self.K3 * observation[0][1] ** 2
            )
        )

        self.action = self.PID_2tank_x1.compute_signal(
            rg.array([[observation[0][0]]]), error_derivative=error_derivative_x1
        ) + self.PID_2tank_x2.compute_signal(
            rg.array([[observation[0][1]]]), error_derivative=error_derivative_x2
        )
        return self.action


class ThreeWheeledRobotDisassembledCLFPolicy(Policy):
    """Nominal scenario for 3-wheel robots used for benchmarking of other scenarios.

    The scenario is sampled.

    For a 3-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here
    a scenario designed by non-smooth backstepping (read more in [[2]_], [[3]_]).

    Attributes:
    m, moment_of_inertia : : numbers
        Mass and moment of inertia around vertical axis of the robot.
    controller_gain : : number
        Controller gain.
        Initial value of the controller's __internal clock.
    sampling_time : : number
        Scenario's sampling time (in seconds).

    References:
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    .. [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    .. [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational asetpointects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(
        self,
        optimizer_config,
        scenario_gain=10,
    ):
        """Initialize an instance of stabilizing policy for three wheeled robot."""
        super().__init__(optimizer_config=optimizer_config)
        from regelum.system import ThreeWheeledRobotDynamic

        self.m = ThreeWheeledRobotDynamic.parameters["m"]
        self.moment_of_inertia = ThreeWheeledRobotDynamic.parameters["I"]
        self.scenario_gain = scenario_gain
        self.xNI_var = self.create_variable(3, 1, name="xNI", is_constant=True)
        self.eta_var = self.create_variable(2, 1, name="eta", is_constant=True)
        self.theta_var = self.create_variable(
            1, 1, name="theta", is_constant=False, like=np.array([0])
        )
        self.register_bounds(self.theta_var, rg.array([[-np.pi, np.pi]]))
        self.register_objective(
            self._Fc, variables=[self.xNI_var, self.eta_var, self.theta_var]
        )

    def _zeta(self, xNI, theta):
        """Compute generic, i.e., theta-dependent, supper_bound_constraintradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)."""
        sigma_tilde = (
            xNI[0] * rg.cos(theta) + xNI[1] * rg.sin(theta) + np.sqrt(rg.abs(xNI[2]))
        )

        nablaF = rg.zeros(3, prototype=theta)

        nablaF[0] = (
            4 * xNI[0] ** 3 - 2 * rg.abs(xNI[2]) ** 3 * rg.cos(theta) / sigma_tilde**3
        )

        nablaF[1] = (
            4 * xNI[1] ** 3 - 2 * rg.abs(xNI[2]) ** 3 * rg.sin(theta) / sigma_tilde**3
        )

        nablaF[2] = (
            (
                3 * xNI[0] * rg.cos(theta)
                + 3 * xNI[1] * rg.sin(theta)
                + 2 * rg.sqrt(rg.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * rg.sign(xNI[2])
            / sigma_tilde**3
        )

        return nablaF

    def _kappa(self, xNI, theta):
        """Stabilizing scenario for NI-part."""
        G = rg.zeros([2, 3], prototype=xNI)
        G[0, :] = rg.hstack([1, 0, xNI[1]])
        G[1, :] = rg.hstack([0, 1, -xNI[0]])
        G = G.T

        kappa_val = rg.zeros(2, prototype=xNI)

        zeta_val = self._zeta(xNI, theta)

        kappa_val[0] = -rg.abs(rg.dot(zeta_val, G[:, 0])) ** (1 / 3) * rg.sign(
            rg.dot(zeta_val, G[:, 0])
        )
        kappa_val[1] = -rg.abs(rg.dot(zeta_val, G[:, 1])) ** (1 / 3) * rg.sign(
            rg.dot(zeta_val, G[:, 1])
        )

        return kappa_val

    def _Fc(self, xNI, eta, theta):
        """Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation."""
        sigma_tilde = (
            xNI[0] * rg.cos(theta) + xNI[1] * rg.sin(theta) + rg.sqrt(rg.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + rg.abs(xNI[2]) ** 3 / sigma_tilde**2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * rg.dot(z, z)

    def _Cart2NH(self, coords_Cart):
        r"""Transform from Cartesian coordinates to non-holonomic (NH) coordinates.

        See Section VIII.A in [[1]_].

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\dot x_3 = x_2 u_1 - x_1 u_2`.

        References:
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """
        xNI = rg.zeros(3)
        eta = rg.zeros(2)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        angle = coords_Cart[2]
        v = coords_Cart[3]
        omega = coords_Cart[4]

        xNI[0] = angle
        xNI[1] = xc * rg.cos(angle) + yc * rg.sin(angle)
        xNI[2] = -2 * (yc * rg.cos(angle) - xc * rg.sin(angle)) - angle * (
            xc * rg.cos(angle) + yc * rg.sin(angle)
        )

        eta[0] = omega
        eta[1] = (yc * rg.cos(angle) - xc * rg.sin(angle)) * omega + v

        return [xNI, eta]

    def _NH2ctrl_Cart(self, xNI, eta, uNI):
        r"""Get control for Cartesian NI from NH coordinates.

        See Section VIII.A in [[1]_].

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\dot x_3 = x_2 u_1 - x_1 u_2`.

        References:
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """
        uCart = rg.zeros(2)

        uCart[0] = self.m * (
            uNI[1]
            + xNI[1] * eta[0] ** 2
            + 1 / 2 * (xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2])
        )
        uCart[1] = self.moment_of_inertia * uNI[0]

        return uCart

    def get_action(self, observation):
        """Perform the same computation as :func:`~Scenario3WRobotDisassembledCLF.compute_action`, but without invoking the __internal clock."""
        observation = observation[0]
        xNI, eta = self._Cart2NH(observation)
        theta_star = self.optimize(xNI=xNI, eta=eta)
        if self.kind == "symbolic":
            theta_star = theta_star["theta"]
        kappa_val = self._kappa(xNI, theta_star)
        z = eta - kappa_val
        uNI = -self.scenario_gain * z
        action = self._NH2ctrl_Cart(xNI, eta, uNI)
        self.action_old = action
        return action.reshape(1, -1)

    def compute_LF(self, observation):
        xNI, eta = self._Cart2NH(observation)
        theta_star = self._minimizer_theta(xNI, eta)
        return self._Fc(xNI, eta, theta_star)
