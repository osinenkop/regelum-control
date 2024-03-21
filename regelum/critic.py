"""Contains critics, which are integrated in scenarios (agents).

Remarks:

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from unittest.mock import MagicMock
import numpy as np
from .utils import rg
from abc import ABC
from .optimizable import Optimizable
from .objective import temporal_difference_objective, temporal_difference_objective_full

try:
    import torch
except (ModuleNotFoundError, ImportError):
    torch = MagicMock()

from .model import Model, ModelNN
from typing import Optional, Union, List
from .optimizable import OptimizerConfig
from .data_buffers import DataBuffer
from .system import System, ComposedSystem
from regelum.typing import RgArray, Weights


class Critic(Optimizable, ABC):
    """Critic base class.

    A critic is an object that estimates or provides the value of a given action or state in a reinforcement learning problem.

    The critic estimates the value of an action by learning from past experience, typically through the optimization of a loss function.
    """

    def __init__(
        self,
        system: Union[System, ComposedSystem],
        model: Union[Model, ModelNN],
        td_n: int = 1,
        device: Union[str, torch.device] = "cpu",
        is_same_critic: bool = False,
        is_value_function: bool = False,
        is_on_policy: bool = False,
        optimizer_config: Optional[OptimizerConfig] = None,
        regularization_param: float = 0.0,
        action_bounds: Optional[Union[List, np.array]] = None,
        size_mesh: Optional[int] = None,
        discount_factor: float = 1.0,
        sampling_time: float = 0.01,
        is_full_iteration_epoch: bool = False,
    ):
        """Initialize a critic object.

        Args:
            system (Union[System, ComposedSystem]): system environment
                that is used in RL problem. The system is mainly used
                for extraction of `dim_observation` and `dim_action`
            model (Union[Model, ModelNN]): Model to use for the critic
            td_n (int, optional): How many running_objective terms to
                use for temporal difference, defaults to 1
            device (Union[str, torch.device], optional): Device to use
                for optimization, defaults to "cpu"
            is_same_critic (bool, optional): whether to use the
                undetached critic in temporal difference loss, defaults
                to False
            is_value_function (bool, optional): whether critis is Value
                function (i.e. depends on observation only) or
                Q-function (i.e. depends on observation and action). For
                `True` the critic is Value function. For `False` the
                critic is Q-function, defaults to False
            is_on_policy (bool, optional): whether critic is on policy
                or off policy optimized, defaults to False
            optimizer_config (Optional[OptimizerConfig], optional):
                optimizer configuration, defaults to None
            regularization_param (bool): L2 penalty for weights displacement
            when symbolic optimization engine is used.
            action_bounds (Optional[Union[List, np.array]], optional):
                action bounds. Needed for DQN algorithm for constraint
                optimization problem, defaults to None
            size_mesh (Optional[int], optional): action grid mesh size,
                needed for DQN algorithm for constraint optimization
                problem, defaults to None
            discount_factor (float, optional): discount factor to use in
                temporal difference loss, defaults to 1.0
            sampling_time (float, optional): scenario sampling time.
                Needed in temporal difference loos, defaults to 0.01
        """
        Optimizable.__init__(self, optimizer_config=optimizer_config)

        self.model = model
        self.discount_factor = discount_factor
        self.system = system
        self.sampling_time = sampling_time
        self.td_n = td_n
        self.device = device
        self.is_same_critic = is_same_critic
        self.is_value_function = is_value_function
        self.is_on_policy = is_on_policy
        self.action_bounds = (
            np.array(action_bounds) if action_bounds is not None else None
        )
        self.regularization_param = regularization_param
        self.size_mesh = size_mesh
        self.is_full_iteration_epoch = is_full_iteration_epoch

        self.initialize_optimize_procedure()

    def receive_estimated_state(self, state):
        self.state = state

    def __call__(
        self, *args, use_stored_weights: bool = False, weights: Optional[Weights] = None
    ) -> Union[RgArray, float]:
        """Compute the value of the critic function for a given observation and/or action.

        Args:
            *args (tuple): tuple of the form (observation, action) or
                (observation,)
            use_stored_weights (bool): flag indicating whether to use
                the stored weights of the critic model or the current
                weights

        Returns:
            Value of the critic function (either value or Q-function)
        """
        return self.model(*args, use_stored_weights=use_stored_weights, weights=weights)

    @property
    def weights(self):
        """Get the weights of the critic model."""
        return self.model.weights

    def update_weights(self, weights: Optional[Weights] = None) -> None:
        """Update the weights of the critic model.

        Args:
            weights: new weights to be used for the critic
                model, if not provided the method does nothing.
        """
        self.model.update_weights(weights)

    def cache_weights(self, weights: Optional[Weights] = None):
        """Store a copy of the current model weights.

        Args:
            weights: An optional ndarray of weights to store. If not
                provided, the current model weights are stored. Default
                is None.
        """
        self.model.cache_weights(weights)

    def restore_weights(self):
        """Restores the model weights to the cached weights."""
        self.model.restore_weights()

    def update_and_cache_weights(self, weights: Optional[Weights] = None):
        """Update the model's weights and cache the new values.

        Args:
            weights: new weights for the model (optional)
        """
        self.update_and_cache_weights(weights)

    def initialize_optimize_procedure(self):
        """Instantilize optimization procedure via Optimizable functionality."""
        self.batch_size = self.get_data_buffer_batch_size()
        (
            self.running_objective_var,
            self.critic_targets_var,
            self.critic_weights_var,
            self.critic_stored_weights_var,
        ) = (
            self.create_variable(
                self.batch_size,
                1,
                name="running_objective",
                is_constant=True,
            ),
            self.create_variable(
                self.batch_size,
                1,
                name="critic_targets",
                is_constant=True,
            ),
            self.create_variable(
                name="critic_weights",
                like=self.model.named_parameters,
            ),
            self.create_variable(
                name="critic_stored_weights",
                like=self.model.cache.weights,
                is_constant=True,
            ),
        )
        if not self.is_value_function:
            self.observation_action_var = self.create_variable(
                self.batch_size,
                self.system.dim_observation + self.system.dim_inputs,
                name="observation_action",
                is_constant=True,
            )
            self.critic_model_output = self.create_variable(
                self.batch_size,
                1,
                name="critic_model_output",
                is_nested_function=True,
                nested_variables=[
                    self.observation_action_var,
                    self.critic_stored_weights_var,
                    self.critic_weights_var,
                ],
            )
        else:
            self.observation_var = self.create_variable(
                self.batch_size,
                self.system.dim_observation,
                name="observation",
                is_constant=True,
            )
            self.critic_model_output = self.create_variable(
                self.batch_size,
                1,
                name="critic_model_output",
                is_nested_function=True,
                nested_variables=[
                    self.observation_var,
                    self.critic_stored_weights_var,
                    self.critic_weights_var,
                ],
            )

        if hasattr(self.model, "weight_bounds"):
            self.register_bounds(self.critic_weights_var, self.model.weight_bounds)

        if self.is_value_function:
            self.connect_source(
                connect_to=self.critic_model_output,
                func=self.model,
                source=self.observation_var,
                weights=self.critic_weights_var,
            )
            if (not self.is_same_critic) and self.is_on_policy:
                self.connect_source(
                    connect_to=self.critic_targets_var,
                    func=self.model.cache,
                    source=self.observation_var,
                    weights=self.critic_stored_weights_var,
                )
        else:
            self.connect_source(
                connect_to=self.critic_model_output,
                func=self.model,
                source=self.observation_action_var,
                weights=self.critic_weights_var,
            )
            if (not self.is_same_critic) and self.is_on_policy:
                self.connect_source(
                    connect_to=self.critic_targets_var,
                    func=self.model.cache,
                    source=self.observation_action_var,
                    weights=self.critic_stored_weights_var,
                )

        self.episode_id_var = self.create_variable(
            self.batch_size,
            1,
            name="episode_id",
            is_constant=True,
        )

        self.register_objective(
            func=self.objective_function,
            variables=[
                self.episode_id_var,
                self.running_objective_var,
                self.critic_model_output,
                self.critic_weights_var,
                self.critic_stored_weights_var,
            ]
            + (
                [self.critic_targets_var]
                if ((not self.is_same_critic) or (not self.is_on_policy))
                else []
            ),
        )

    def data_buffer_objective_keys(self) -> List[str]:
        """Return a list of `regelum.data_buffers.DataBuffer` keys to be used for the substitution to the objective function.

        Returns:
            List of keys.
        """
        if self.is_value_function:
            keys = ["observation", "running_objective", "episode_id"]
        else:
            keys = ["observation_action", "running_objective", "episode_id"]

        if not self.is_on_policy:
            keys.append("critic_targets")

        return keys

    def regularization_function(self, critic_stored_weights, critic_weights):
        return self.regularization_param * rg.sum(
            (critic_stored_weights - critic_weights) ** 2
        )

    def objective_function(
        self,
        critic_model_output,
        running_objective,
        critic_stored_weights,
        critic_weights,
        episode_id,
        critic_targets=None,
    ):
        if self.is_full_iteration_epoch:
            assert (
                self.kind == "tensor"
            ), "Full iteration epoch only supported for Torch critic"
            td_objective = temporal_difference_objective_full(
                critic_model_output=critic_model_output,
                running_objective=running_objective,
                episode_ids=episode_id,
                td_n=self.td_n,
                discount_factor=self.discount_factor,
                sampling_time=self.sampling_time,
                critic_targets=critic_targets,
            )
        else:
            td_objective = temporal_difference_objective(
                critic_model_output=critic_model_output,
                running_objective=running_objective,
                td_n=self.td_n,
                discount_factor=self.discount_factor,
                sampling_time=self.sampling_time,
                critic_targets=critic_targets,
            )
        if self.kind == "tensor":
            return td_objective

        else:
            return td_objective + self.regularization_function(
                critic_stored_weights=critic_stored_weights,
                critic_weights=critic_weights,
            )

    def update_data_buffer_with_optimal_policy_targets(self, data_buffer: DataBuffer):
        assert self.size_mesh is not None, "Specify size_mesh for off-policy critic"
        data_buffer_size = len(data_buffer)
        if data_buffer_size == 0:
            return
        observations = np.vstack(data_buffer.data["observation"])
        actions = np.random.uniform(
            self.action_bounds[:, 0],
            self.action_bounds[:, 1],
            size=(data_buffer_size, self.size_mesh, len(self.action_bounds)),
        )
        tiled_observations = np.vstack(
            [
                [np.tile(observation, (self.size_mesh, 1))]
                for observation in observations
            ]
        )
        model_inputs = torch.FloatTensor(
            np.concatenate((tiled_observations, actions), axis=2)
        ).to(self.device)

        values = self.model(model_inputs, use_stored_weights=True)
        data_buffer.update({"critic_targets": torch.min(values, axis=1).values.numpy()})

    def update_data_buffer_with_stored_weights_critic_output(
        self, data_buffer: DataBuffer
    ):
        self.model(data_buffer, use_stored_weights=True)

    def delete_critic_targets(self, data_buffer: DataBuffer):
        if "critic_targets" in data_buffer.keys():
            data_buffer.delete_key("critic_targets")

    def optimize(
        self, data_buffer, is_update_and_cache_weights=True, is_constrained=True
    ):
        if isinstance(self.model, ModelNN):
            self.model.to(self.device)
            self.model.cache.to(self.device)

        if not self.is_on_policy:
            self.update_data_buffer_with_optimal_policy_targets(data_buffer)

        opt_kwargs = data_buffer.get_optimization_kwargs(
            keys=self.data_buffer_objective_keys(),
            optimizer_config=self.optimizer_config,
        )
        weights = None
        if opt_kwargs is not None:
            if self.kind == "tensor":
                super().optimize(**opt_kwargs, is_constrained=is_constrained)
                if is_update_and_cache_weights:
                    self.model.update_and_cache_weights()
            elif self.kind == "symbolic":
                weights = super().optimize(
                    **opt_kwargs,
                    is_constrained=is_constrained,
                    critic_weights=self.model.weights,
                    critic_stored_weights=self.model.cache.weights,
                )["critic_weights"]
                if is_update_and_cache_weights:
                    self.model.update_and_cache_weights(weights)

        if not self.is_on_policy:
            self.delete_critic_targets(data_buffer)

        return weights


class CriticTrivial(Critic):
    """A mocked Critic object."""

    class TrivialModel:
        """A mocked Trivial model."""

        def named_parameters(self):
            return None

    def __init__(self, *args, **kwargs):
        """Instantiate a CriticTrivial object."""
        self.model = self.TrivialModel()

    def optimize(self, *args, **kwargs):
        pass

    def update_weights(self, *args, **kwargs):
        pass

    def cache_weights(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    @property
    def weights(self):
        return None


class CriticCALF(Critic):
    """Critic for CALF algorithm."""

    def __init__(
        self,
        system,
        model: Union[Model, ModelNN],
        is_same_critic: bool,
        is_value_function: bool,
        td_n: int = 1,
        predictor: Optional[Model] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        discount_factor: float = 1.0,
        sampling_time: float = 0.01,
        ######
        safe_decay_param: float = 1e-4,
        is_dynamic_decay_rate: bool = True,
        safe_policy=None,
        lb_parameter: float = 1e-6,
        ub_parameter: float = 1e3,
        regularization_param: float = 0,
    ):
        """Instantiate a CriticCALF object."""
        super().__init__(
            system=system,
            model=model,
            td_n=td_n,
            is_same_critic=is_same_critic,
            is_value_function=is_value_function,
            is_on_policy=True,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            action_bounds=None,
            regularization_param=regularization_param,
        )
        self.predictor = predictor
        self.safe_decay_param = safe_decay_param
        self.is_dynamic_decay_rate = is_dynamic_decay_rate
        if not self.is_dynamic_decay_rate:
            self.safe_decay_rate = self.safe_decay_param

        self.lb_parameter = lb_parameter
        self.ub_parameter = ub_parameter
        self.safe_policy = safe_policy

        self.observation_last_good_var = self.create_variable(
            self.batch_size,
            self.system.dim_observation,
            name="observation_last_good",
            is_constant=True,
        )
        self.prev_good_critic_var = self.create_variable(
            1,
            1,
            name="prev_good_critic",
            is_constant=True,
            is_nested_function=True,
            nested_variables=[
                self.observation_last_good_var,
                self.critic_stored_weights_var,
            ],
        )
        self.connect_source(
            connect_to=self.prev_good_critic_var,
            func=self.model.cache,
            source=self.observation_last_good_var,
            weights=self.critic_stored_weights_var,
        )

        self.register_constraint(
            self.CALF_decay_constraint_no_prediction,
            variables=[
                self.critic_model_output,
                self.prev_good_critic_var,
            ],
        )
        self.register_constraint(
            self.CALF_critic_lower_bound_constraint,
            variables=[self.critic_model_output, self.observation_var],
        )

        self.observation_last_good = None

    def data_buffer_objective_keys(self) -> List[str]:
        """Return a list of `regelum.data_buffers.DataBuffer` keys to be used for the substitution to the objective function.

        Returns:
            List of keys.
        """
        keys = super().data_buffer_objective_keys()
        keys.append("observation_last_good")
        return keys

    def CALF_decay_constraint_no_prediction(
        self,
        critic_model_output,
        prev_good_critic,
    ):
        stabilizing_constraint_violation = (
            critic_model_output[-1, :]
            - prev_good_critic[-2, :]
            + self.sampling_time * self.safe_decay_rate
        )
        return stabilizing_constraint_violation

    def CALF_critic_lower_bound_constraint(
        self, critic_model_output: RgArray, observation: RgArray
    ) -> RgArray:
        """Constraint that ensures that the value of the critic is above a certain lower bound.

        The lower bound is determined by the `current_observation` and a certain constant.

        Args:
            critic_model_output: output of a critic

        Returns:
            Constraint violation
        """
        self.lb_constraint_violation = (
            self.lb_parameter * rg.sum(observation[:-1, :] ** 2)
            - critic_model_output[-1, :]
        )
        return self.lb_constraint_violation
