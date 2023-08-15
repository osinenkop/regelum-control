"""Contains critics, which are integrated in controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import rcognita


import numpy as np
from .__utilities import rc, NUMPY, CASADI, TORCH, Clock
from abc import ABC
import scipy as sp
import random
from .optimizable import Optimizable
from .objective import temporal_difference_objective

try:
    import torch
except (ModuleNotFoundError, ImportError):
    from unittest.mock import MagicMock

    torch = MagicMock()

from .model import ModelWeightContainer
from .model import Model, ModelNN
from .objective import Objective
from typing import Optional, Union, List
from .optimizable import OptimizerConfig
from .data_buffers import DataBuffer


class Critic(Optimizable, ABC):
    """Critic base class.

    A critic is an object that estimates or provides the value of a given action or state in a reinforcement learning problem.

    The critic estimates the value of an action by learning from past experience, typically through the optimization of a loss function.
    """

    def __init__(
        self,
        system,
        model: Optional[Model] = None,
        running_objective: Optional[Objective] = None,
        td_n: int = 1,
        device: Union[str, torch.device] = "cpu",
        is_same_critic: bool = False,
        is_value_function: bool = False,
        is_on_policy: bool = False,
        optimizer_config: Optional[OptimizerConfig] = None,
        action_bounds: Optional[Union[List, np.array]] = None,
        size_mesh: Optional[int] = None,
        discount_factor: float = 1.0,
        sampling_time: float = 0.01,
    ):
        """Initialize a critic object.

        :param optimizer: Optimizer to use for training the critic
        :type optimizer: Optional[Optimizer]
        :param model: Model to use for the critic
        :type model: Optional[Model]
        :param running_objective: Objective function to use for the critic
        :type running_objective: Optional[Objective]
        :param discount_factor: Discount factor to use in the value calculation
        :type discount_factor: float
        :param sampling_time: Sampling time for the critic
        :type sampling_time: float
        :param critic_regularization_param: Regularization parameter for the critic
        :type critic_regularization_param: float
        """
        Optimizable.__init__(self, optimizer_config=optimizer_config)
        if model:
            self.model = model
        else:
            raise ValueError("No model defined")

        self.discount_factor = discount_factor
        self.running_objective = running_objective
        self.system = system
        self.current_critic_loss = 0
        self.total_objective = 0
        self.sampling_time = sampling_time
        self.td_n = td_n
        self.device = device
        self.is_same_critic = is_same_critic
        self.is_value_function = is_value_function
        self.is_on_policy = is_on_policy
        self.action_bounds = (
            np.array(action_bounds) if action_bounds is not None else None
        )
        self.size_mesh = size_mesh

        self.initialize_optimize_procedure()

    def receive_state(self, state):
        self.state = state

    def __call__(self, *args, use_stored_weights=False, weights=None):
        """Compute the value of the critic function for a given observation and/or action.

        :param args: tuple of the form (observation, action) or (observation,)
        :type args: tuple
        :param use_stored_weights: flag indicating whether to use the stored weights of the critic model or the current weights
        :type use_stored_weights: bool
        :return: value of the critic function
        :rtype: float
        """
        return self.model(*args, use_stored_weights=use_stored_weights, weights=weights)

    @property
    def weights(self):
        """Get the weights of the critic model."""
        return self.model.weights

    def update_weights(self, weights=None):
        """Update the weights of the critic model.

        :param weights: new weights to be used for the critic model, if not provided the optimized weights will be used
        :type weights: numpy array
        """
        if weights is None:
            self.model.update_weights(self.optimized_weights)
        else:
            self.model.update_weights(weights)

    def cache_weights(self, weights=None):
        """Store a copy of the current model weights.

        :param weights: An optional ndarray of weights to store. If not provided, the current
            model weights are stored. Default is None.
        """
        if weights is not None:
            self.model.cache_weights(weights)
        else:
            self.model.cache_weights(self.optimized_weights)

    def restore_weights(self):
        """Restores the model weights to the cached weights."""
        self.model.restore_weights()

    def update_and_cache_weights(self, weights=None):
        """Update the model's weights and cache the new values.

        :param weights: new weights for the model (optional)
        """
        self.update_weights(weights)
        self.cache_weights(weights)

    def accept_or_reject_weights(
        self, weights, constraint_functions=None, optimizer_engine="SciPy", atol=1e-10
    ):
        """Determine whether to accept or reject the given weights based on whether they violate the given constraints.

        Normally, this method takes weights and checks CALF constraints by plugging them into the critic model.
        This works in a straightforward way with scipy and CASADi optimizers.
        In case of Torch, the weights are stored in the model after the learning.
        So, we can simply call CALF constraints directly on the trained Torch model.
        But to keep the method signature the same, we formally take weights as an input argument.
        See the optimizer checking condition in the code of this method.

        :param weights: weights to evaluate
        :type weights: numpy array
        :param constraint_functions: functions that return the constraint violations for the given weights
        :type constraint_functions: list of functions
        :param optimizer_engine: optimizer engine used
        :type optimizer_engine: str
        :param atol: absolute tolerance for the constraints (default is 1e-10)
        :type atol: float
        :return: string indicating whether the weights were accepted or rejected
        :rtype: str
        """
        if constraint_functions is None:
            constraints_not_violated = True
        else:
            if self.optimizer_engine != rc.TORCH:
                not_violated = [cond(weights) <= atol for cond in constraint_functions]
                constraints_not_violated = all(not_violated)
            else:
                not_violated = [cond() <= atol for cond in constraint_functions]
                constraints_not_violated = all(not_violated)
            # print(not_violated)

        if constraints_not_violated:
            return "accepted"
        else:
            return "rejected"

    def update_total_objective(self, observation, action):
        """Update the outcome variable based on the running objective and the current observation and action.

        :param observation: current observation
        :type observation: np.ndarray
        :param action: current action
        :type action: np.ndarray.
        """
        self.total_objective += (
            self.running_objective(observation, action) * self.sampling_time
        )

    def reset(self):
        """Reset the outcome and current critic loss variables, and re-initialize the buffers."""
        self.total_objective = 0
        self.current_critic_loss = 0

    def initialize_optimize_procedure(self):
        batch_size = self.get_data_buffer_batch_size()
        (
            self.running_objective_var,
            self.observation_var,
            self.observation_action_var,
            self.critic_targets_var,
            self.critic_model_output,
            self.critic_weights_var,
            self.critic_stored_weights_var,
        ) = (
            self.create_variable(
                batch_size,
                1,
                name="running_objective",
                is_constant=True,
            ),
            self.create_variable(
                batch_size,
                self.system.dim_observation,
                name="observation",
                is_constant=True,
            ),
            self.create_variable(
                batch_size,
                self.system.dim_observation + self.system.dim_inputs,
                name="observation_action",
                is_constant=True,
            ),
            self.create_variable(
                batch_size,
                1,
                name="critic_targets",
                is_constant=True,
            ),
            self.create_variable(batch_size, 1, name="critic_model_output"),
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

        self.register_objective(
            func=self.objective_function,
            variables=[
                self.running_objective_var,
                self.critic_model_output,
            ]
            + (
                [self.critic_targets_var]
                if ((not self.is_same_critic) or (not self.is_on_policy))
                else []
            ),
        )

    def data_buffer_objective_keys(self) -> List[str]:
        if self.is_value_function:
            keys = ["observation", "running_objective"]
        else:
            keys = ["observation_action", "running_objective"]

        if not self.is_on_policy:
            keys.append("critic_targets")

        return keys

    def objective_function(
        self,
        critic_model_output,
        running_objective,
        critic_targets=None,
    ):
        return temporal_difference_objective(
            critic_model_output=critic_model_output,
            running_objective=running_objective,
            td_n=self.td_n,
            discount_factor=self.discount_factor,
            sampling_time=self.sampling_time,
            critic_targets=critic_targets,
        )

    def update_data_buffer_with_optimal_policy_targets(self, data_buffer: DataBuffer):
        assert self.size_mesh is not None, "Specify size_mesh for off-policy critic"
        data_buffer_size = len(data_buffer)
        if data_buffer_size == 0:
            return
        observations = data_buffer.getitem(
            slice(0, data_buffer_size), keys=["observation"], dtype=np.array
        )["observation"]
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

    def optimize_on_event(self, data_buffer):
        if isinstance(self.model, ModelNN):
            self.model.to(self.device)
            self.model.cache.to(self.device)

        if not self.is_on_policy:
            self.update_data_buffer_with_optimal_policy_targets(data_buffer)

        opt_kwargs = data_buffer.get_optimization_kwargs(
            keys=self.data_buffer_objective_keys(),
            optimizer_config=self.optimizer_config,
        )

        if opt_kwargs is not None:
            if self.kind == "tensor":
                weights = self.optimize(
                    **opt_kwargs,
                )
            elif self.kind == "symbolic":
                weights = self.optimize(
                    **opt_kwargs,
                    critic_weights=self.model.weights,
                    critic_stored_weights=self.model.cache.weights,
                )
            if isinstance(weights, dict):
                weights = weights["critic_weights"]
                self.model.update_and_cache_weights(weights)
            else:
                self.model.update_and_cache_weights()

        if not self.is_on_policy:
            self.delete_critic_targets(data_buffer)


class CriticTrivial(Critic):
    """A mocked Critic object."""

    def __init__(self, *args, **kwargs):
        from unittest.mock import MagicMock

        self.model = MagicMock()

    def optimize_on_event(self, *args, **kwargs):
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


class CriticCALF:
    def __init__(
        self,
        *args,
        safe_decay_param=1e-3,
        is_dynamic_decay_rate=True,
        predictor=None,
        state_init=None,
        safe_controller=None,
        penalty_param=0,
        is_predictive=True,
        action_init=None,
        lb_parameter=1e-6,
        ub_parameter=1e3,
        **kwargs,
    ):
        """Initialize a CriticCALF object.

        :param args: Arguments to be passed to the base class `CriticOfObservation`.
        :param safe_decay_param: Rate at which the safe set shrinks over time.
        :param is_dynamic_decay_rate: Whether the decay rate should be dynamic or not.
        :param predictor: A predictor object to be used to predict future observations.
        :param observation_init: Initial observation to be used to initialize the safe set.
        :param safe_controller: Safe controller object to be used to compute stabilizing actions.
        :param penalty_param: Penalty parameter to be used in the CALF objective.
        :param is_predictive: Whether the safe constraints should be computed based on predictions or not.
        :param kwargs: Keyword arguments to be passed to the base class `CriticOfObservation`.
        """
        super().__init__(*args, **kwargs)
        self.safe_decay_param = safe_decay_param
        self.is_dynamic_decay_rate = is_dynamic_decay_rate
        if not self.is_dynamic_decay_rate:
            self.safe_decay_rate = self.safe_decay_param

        self.lb_parameter = lb_parameter
        self.ub_parameter = ub_parameter
        self.safe_controller = safe_controller
        self.predictor = predictor
        self.observation_init = self.predictor.system.get_observation(
            time=0, state=state_init, inputs=action_init
        )
        self.action_init = action_init
        self.observation_last_good = self.observation_init
        self.r_prev_init = self.r_prev = self.running_objective(
            self.observation_init, self.action_init
        )

        self.lb_constraint_violations = []
        self.ub_constraint_violations = []
        self.stabilizing_constraint_violations = []
        self.values = []
        self.times = []
        self.Ls = []
        self.CALFs = []
        self.penalty_param = penalty_param
        self.expected_CALFs = []
        self.stabilizing_constraint_violation = 0
        self.CALF = 0
        self.weights_acceptance_status = False

        if is_predictive:
            self.CALF_decay_constraint = (
                self.CALF_decay_constraint_predicted_safe_policy
            )
            self.CALF_critic_lower_bound_constraint = (
                self.CALF_critic_lower_bound_constraint_predictive
            )
            # self.CALF_decay_constraint = self.CALF_decay_constraint_predicted_on_policy
        else:
            self.CALF_decay_constraint = self.CALF_decay_constraint_no_prediction

        self.intrinsic_constraints = [
            self.CALF_decay_constraint,
            # self.CALF_critic_lower_bound_constraint,
        ]

    def reset(self):
        """Reset the critic to its initial state."""
        super().reset()
        self.observation_last_good = self.observation_init
        self.r_prev = self.r_prev_init

        if hasattr(self.safe_controller, "reset_all_PID_controllers"):
            self.safe_controller.reset_all_PID_controllers()

    def update_buffers(self, observation, action):
        """Update data buffers and dynamic safe decay rate.

        Updates the observation and action data buffers with the given observation and action.
        Updates the outcome using the given observation and action.
        Updates the current observation and action with the given observation and action.
        If the flag is_dynamic_decay_rate is set to True, also updates the safe decay rate with the L2 norm of the given observation.

        :param observation: The new observation to be added to the observation buffer.
        :type observation: numpy.ndarray
        :param action: The new action to be added to the action buffer.
        :type action: numpy.ndarray
        """
        self.action_buffer = rc.push_vec(
            self.action_buffer, rc.array(action, prototype=self.action_buffer)
        )
        self.observation_buffer = rc.push_vec(
            self.observation_buffer,
            rc.array(observation, prototype=self.observation_buffer),
        )
        self.update_total_objective(observation, action)

        self.current_observation = observation
        self.current_action = action

        if self.is_dynamic_decay_rate:
            # print(self.safe_decay_param)
            self.safe_decay_rate = self.safe_decay_param * rc.norm_2(observation)
            # self.safe_decay_param = self.safe_deay_rate_param * rc norm ...

    def CALF_decay_constraint_no_prediction(self, weights=None):
        """Constraint that ensures that the CALF value is decreasing by a certain rate.

        The rate is determined by the `safe_decay_param` parameter. This constraint is used when there is no prediction of the next state.
        :param weights: critic weights to be evaluated
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """

        critic_curr = self.model(self.current_observation, weights=weights)
        critic_prev = self.model(
            self.observation_last_good,
            use_stored_weights=True,
        )

        self.stabilizing_constraint_violation = (
            critic_curr
            - critic_prev
            + self.predictor.pred_step_size * self.safe_decay_rate
        )
        return self.stabilizing_constraint_violation

    def CALF_critic_lower_bound_constraint(self, weights=None):
        """Constraint that ensures that the value of the critic is above a certain lower bound.

        The lower bound is determined by the `current_observation` and a certain constant.
        :param weights: critic weights to be evaluated
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """
        self.lb_constraint_violation = self.lb_parameter * rc.norm_2(
            self.current_observation
        ) - self.model(self.current_observation, weights=weights)
        return self.lb_constraint_violation

    def CALF_critic_lower_bound_constraint_predictive(self, weights=None):
        """Constraint that ensures that the value of the critic is above a certain lower bound.

        The lower bound is determined by
        the `current_observation` and a certain constant.
        :param weights: critic weights to be evaluated
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """
        action = self.safe_controller.compute_action(self.current_observation)
        predicted_observation = self.predictor.system.get_observation(
            time=None, state=self.predictor.predict(self.state, action), inputs=action
        )
        self.lb_constraint_violation = self.lb_parameter * rc.norm_2(
            predicted_observation
        ) - self.model(predicted_observation, weights=weights)
        return self.lb_constraint_violation

    def CALF_critic_upper_bound_constraint(self, weights=None):
        """Calculate the constraint violation for the CALF decay constraint when no prediction is made.

        :param weights: critic weights
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """
        self.ub_constraint_violation = self.model(
            self.current_observation, weights=weights
        ) - self.ub_parameter * rc.norm_2(self.current_observation)
        return self.ub_constraint_violation

    def CALF_decay_constraint_predicted_safe_policy(self, weights=None):
        """Calculate the constraint violation for the CALF decay constraint when a predicted safe policy is used.

        :param weights: critic weights
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """
        observation_last_good = self.observation_last_good

        self.safe_action = action = self.safe_controller.compute_action(
            self.current_observation
        )
        self.predicted_observation = (
            predicted_observation
        ) = self.predictor.system.get_observation(
            time=None, state=self.predictor.predict(self.state, action), inputs=action
        )

        self.critic_next = self.model(predicted_observation, weights=weights)
        self.critic_current = self.model(observation_last_good, use_stored_weights=True)

        self.stabilizing_constraint_violation = (
            self.critic_next
            - self.critic_current
            + self.predictor.pred_step_size * self.safe_decay_param
        )
        return self.stabilizing_constraint_violation

    def CALF_decay_constraint_predicted_on_policy(self, weights=None):
        """Constraint for ensuring that the CALF function decreases at each iteration.

        This constraint is used when prediction is done using the last action taken.

        :param weights: Current weights of the critic network.
        :type weights: ndarray
        :return: Violation of the constraint. A positive value indicates violation.
        :rtype: float
        """
        action = self.action_buffer[:, -1]
        predicted_observation = self.predictor.system.get_observation(
            time=None, state=self.predictor.predict(self.state, action), inputs=action
        )
        self.stabilizing_constraint_violation = (
            self.model(predicted_observation, weights=weights)
            - self.model(
                self.observation_last_good,
                use_stored_weights=True,
            )
            + self.predictor.pred_step_size * self.safe_decay_param
        )
        return self.stabilizing_constraint_violation

    def objective(self, data_buffer=None, weights=None):
        """Objective of the critic, say, a squared temporal difference."""
        if data_buffer is None:
            observation_buffer = self.observation_buffer
            action_buffer = self.action_buffer
        else:
            observation_buffer = data_buffer["observation_buffer"]
            action_buffer = data_buffer["action_buffer"]

        critic_objective = 0

        for k in range(self.data_buffer_size - 1, 0, -1):
            observation_old = observation_buffer[:, k - 1]
            observation_next = observation_buffer[:, k]
            action_next = action_buffer[:, k - 1]

            # Temporal difference

            critic_old = self.model(observation_old, weights=weights)
            critic_next = self.model(observation_next, use_stored_weights=True)

            if self.critic_regularization_param > 1e-9:
                weights_current = weights
                weights_last_good = self.model.cache.weights
                regularization_term = (
                    rc.sum_2(weights_current - weights_last_good)
                    * self.critic_regularization_param
                )
            else:
                regularization_term = 0

            temporal_difference = (
                critic_old
                - self.discount_factor * critic_next
                - self.running_objective(observation_next, action_next)
            )

            critic_objective += 1 / 2 * temporal_difference**2 + regularization_term

        if self.intrinsic_constraints != [] and self.penalty_param > 0:
            for constraint in self.intrinsic_constraints:
                critic_objective += self.penalty_param * rc.penalty_function(
                    constraint(), penalty_coeff=1.0e-1
                )

        return critic_objective
