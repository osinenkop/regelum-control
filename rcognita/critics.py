"""
This module containing critics, which are integrated in controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
import numpy as np
from .__utilities import rc, NUMPY, CASADI, TORCH, Clock
from abc import ABC, abstractmethod
import scipy as sp
from functools import partial
import torch
from copy import deepcopy
from multiprocessing import Pool
from .models import ModelWeightContainer
from .optimizers import Optimizer
from .models import Model
from .objectives import Objective
from typing import Optional, Union


class Critic(ABC):
    """
    Critic base class.

    A critic is an object that estimates or provides the value of a given action or state in a reinforcement learning problem.

    The critic estimates the value of an action by learning from past experience, typically through the optimization of a loss function.
    """

    def __init__(
        self,
        system_dim_input: int,
        system_dim_output: int,
        data_buffer_size: int,
        optimizer: Optional[Optimizer] = None,
        model: Optional[Model] = None,
        running_objective: Optional[Objective] = None,
        discount_factor: float = 1.0,
        observation_target: Optional[np.ndarray] = None,
        sampling_time: float = 0.01,
        critic_regularization_param: float = 0.0,
    ):
        """
        Initialize a critic object.

        :param system_dim_input: Dimension of the input data
        :type system_dim_input: int
        :param system_dim_output: Dimension of the output data
        :type system_dim_output: int
        :param data_buffer_size: Maximum size of the data buffer
        :type data_buffer_size: int
        :param optimizer: Optimizer to use for training the critic
        :type optimizer: Optional[Optimizer]
        :param model: Model to use for the critic
        :type model: Optional[Model]
        :param running_objective: Objective function to use for the critic
        :type running_objective: Optional[Objective]
        :param discount_factor: Discount factor to use in the value calculation
        :type discount_factor: float
        :param observation_target: Target observation for the critic
        :type observation_target: Optional[np.ndarray]
        :param sampling_time: Sampling time for the critic
        :type sampling_time: float
        :param critic_regularization_param: Regularization parameter for the critic
        :type critic_regularization_param: float
        """
        self.data_buffer_size = data_buffer_size
        self.system_dim_input = system_dim_input
        self.system_dim_output = system_dim_output

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            raise ValueError("No optimizer defined")

        if model:
            self.model = model
        else:
            raise ValueError("No model defined")

        self.initialize_buffers()

        if observation_target is None:
            observation_target = np.array([])

        self.observation_target = observation_target

        self.discount_factor = discount_factor
        self.running_objective = running_objective

        self.current_critic_loss = 0
        self.outcome = 0
        self.sampling_time = sampling_time
        self.clock = Clock(sampling_time)
        self.intrinsic_constraints = []
        self.penalty_param = 0
        self.critic_regularization_param = critic_regularization_param

    @property
    def optimizer_engine(self):
        """Returns the engine used by the optimizer.
        
        :return: A string representing the engine used by the optimizer.
            Can be one of 'Torch', 'CasADi', or 'Numpy'.
        """
        if self.optimizer.engine == "Torch":
            return TORCH
        elif self.optimizer.engine == "CasADi":
            return CASADI
        else:
            return NUMPY

    def __call__(self, *args, use_stored_weights=False):
        """
        Compute the value of the critic function for a given observation and/or action.
        
        :param args: tuple of the form (observation, action) or (observation,)
        :type args: tuple
        :param use_stored_weights: flag indicating whether to use the stored weights of the critic model or the current weights
        :type use_stored_weights: bool
        :return: value of the critic function
        :rtype: float
        """
        if len(args) == 2:
            chi = rc.concatenate(tuple(args))
        else:
            chi = args[0]
        return self.model(chi, use_stored_weights=use_stored_weights)

    def update_weights(self, weights=None):
        """
        Update the weights of the critic model.
        
        :param weights: new weights to be used for the critic model, if not provided the optimized weights will be used
        :type weights: numpy array
        """
        if weights is None:
            self.model.update_weights(self.optimized_weights)
        else:
            self.model.update_weights(weights)

    def cache_weights(self, weights=None):
        """
        Stores a copy of the current model weights.
        
        :param weights: An optional ndarray of weights to store. If not provided, the current
            model weights are stored. Default is None.
        """
        if weights is not None:
            self.model.cache_weights(weights)
        else:
            self.model.cache_weights(self.optimized_weights)

    def restore_weights(self):
        """
        Restores the model weights to the cached weights.
        """
        self.model.restore_weights()

    def update_and_cache_weights(self, weights=None):
        """
        Update the model's weights and cache the new values.

        :param weights: new weights for the model (optional)
        """
        self.update_weights(weights)
        self.cache_weights(weights)

    def accept_or_reject_weights(
        self, weights, constraint_functions=None, optimizer_engine="SciPy", atol=1e-10
    ):
        """
        Determine whether to accept or reject the given weights based on whether they violate the given constraints.

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
            not_violated = [cond(weights) <= atol for cond in constraint_functions]
            constraints_not_violated = all(not_violated)
            print(not_violated)

        if constraints_not_violated:
            return "accepted"
        else:
            return "rejected"

    def optimize_weights(
        self, time=None,
    ):
        """
        Compute optimized critic weights, possibly subject to constraints.
        If weights satisfying constraints are found, the method returns the status `accepted`.
        Otherwise, it returns the status `rejected`.

        :param time: optional time parameter for use in CasADi and SciPy optimization.
        :type time: float, optional
        :return: acceptance status of the optimized weights, either `accepted` or `rejected`.
        :rtype: str
        """
        if self.optimizer.engine == "CasADi":
            self.optimized_weights = self._CasADi_update(self.intrinsic_constraints)

        elif self.optimizer.engine == "SciPy":
            self.optimized_weights = self._SciPy_update(self.intrinsic_constraints)

        elif self.optimizer.engine == "Torch":
            self._Torch_update()
            self.optimized_weights = self.model.weights

        if self.intrinsic_constraints:
            # print("with constraint functions")
            self.weights_acceptance_status = self.accept_or_reject_weights(
                self.optimized_weights,
                optimizer_engine=self.optimizer.engine,
                constraint_functions=self.intrinsic_constraints,
            )
        else:
            # print("without constraint functions")
            self.weights_acceptance_status = "accepted"

        return self.weights_acceptance_status

    def update_buffers(self, observation, action):
        """
        Updates the buffers of the critic with the given observation and action.

        :param observation: the current observation of the system.
        :type observation: np.ndarray
        :param action: the current action taken by the actor.
        :type action: np.ndarray
        """
        self.action_buffer = rc.push_vec(
            self.action_buffer, rc.array(action, prototype=self.action_buffer)
        )
        self.observation_buffer = rc.push_vec(
            self.observation_buffer,
            rc.array(observation, prototype=self.observation_buffer),
        )
        self.update_outcome(observation, action)

        self.current_observation = observation
        self.current_action = action

    def initialize_buffers(self):
        """
        Initialize the action and observation buffers with zeros.
        """
        self.action_buffer = rc.zeros(
            (int(self.system_dim_input), int(self.data_buffer_size)),
            rc_type=self.optimizer_engine,
        )
        self.observation_buffer = rc.zeros(
            (int(self.system_dim_output), int(self.data_buffer_size)),
            rc_type=self.optimizer_engine,
        )

    def update_outcome(self, observation, action):
        """
        Update the outcome variable based on the running objective and the current observation and action.
        :param observation: current observation
        :type observation: np.ndarray
        :param action: current action
        :type action: np.ndarray
        """

        self.outcome += self.running_objective(observation, action) * self.sampling_time

    def reset(self):
        """
        Reset the outcome and current critic loss variables, and re-initialize the buffers.
        """
        self.outcome = 0
        self.current_critic_loss = 0
        self.initialize_buffers()

    def _SciPy_update(self, intrinsic_constraints=None):

        weights_init = self.model.cache.weights

        constraints = ()
        weight_bounds = [self.model.weight_min, self.model.weight_max]
        data_buffer = {
            "observation_buffer": self.observation_buffer,
            "action_buffer": self.action_buffer,
        }

        cost_function = lambda weights: self.objective(data_buffer, weights=weights)
        is_penalty = int(self.penalty_param > 0)
        if intrinsic_constraints:
            constraints = tuple(
                [
                    sp.optimize.NonlinearConstraint(constraint, -np.inf, 0.0)
                    for constraint in intrinsic_constraints[is_penalty:]
                ]
            )

        optimized_weights = self.optimizer.optimize(
            cost_function, weights_init, weight_bounds, constraints=constraints,
        )
        return optimized_weights

    def _CasADi_update(self, intrinsic_constraints=None):

        weights_init = rc.DM(self.model.cache.weights)
        symbolic_var = rc.array_symb(tup=rc.shape(weights_init), prototype=weights_init)

        constraints = ()
        weight_bounds = [self.model.weight_min, self.model.weight_max]
        data_buffer = {
            "observation_buffer": self.observation_buffer,
            "action_buffer": self.action_buffer,
        }

        cost_function = lambda weights: self.objective(data_buffer, weights=weights)

        cost_function = rc.lambda2symb(cost_function, symbolic_var)

        is_penalty = int(self.penalty_param > 0)

        if intrinsic_constraints:
            constraints = [
                rc.lambda2symb(constraint, symbolic_var)
                for constraint in intrinsic_constraints[is_penalty:]
            ]

        optimized_weights = self.optimizer.optimize(
            cost_function,
            weights_init,
            weight_bounds,
            constraints=constraints,
            decision_variable_symbolic=symbolic_var,
        )

        self.cost_function = cost_function
        self.constraint = constraints
        self.weights_init = weights_init
        self.symbolic_var = symbolic_var

        return optimized_weights

    def _Torch_update(self):

        data_buffer = {
            "observation_buffer": torch.tensor(self.observation_buffer),
            "action_buffer": torch.tensor(self.action_buffer),
        }

        self.optimizer.optimize(
            objective=self.objective, model=self.model, model_input=data_buffer,
        )

        self.current_critic_loss = self.objective(data_buffer).detach().numpy()

    def update_target(self, new_target):
        self.target = new_target

    @abstractmethod
    def objective(self):
        pass


class CriticOfObservation(Critic):
    """
    This is the class of critics that are represented as functions of observation only.
    """

    def objective(self, data_buffer=None, weights=None):
        """
        Objective of the critic, say, a squared temporal difference.

        """
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
            action_old = action_buffer[:, k - 1]

            # Temporal difference

            critic_old = self.model(observation_old, weights=weights)
            critic_next = self.model(observation_next, use_stored_weights=True)

            weights_current = weights
            weights_last_good = self.model.cache.weights
            if self.critic_regularization_param > 0:
                regularization_term = (
                    rc.sum_2(weights_current - weights_last_good)
                    * self.critic_regularization_param
                )
            else:
                regularization_term = 0

            temporal_difference = (
                critic_old
                - self.discount_factor * critic_next
                - self.running_objective(observation_old, action_old)
            )

            critic_objective += 1 / 2 * temporal_difference ** 2 + regularization_term

        return critic_objective


class CriticOfActionObservation(Critic):
    """
    This is the class of critics that are represented as functions of observation only.
    """

    def objective(self, data_buffer=None, weights=None):
        """
        Compute the objective function of the critic, which is typically a squared temporal difference.

        :param data_buffer: a dictionary containing the action and observation buffers, if different from the class attributes.
        :type data_buffer: dict, optional
        :param weights: the weights of the critic model, if different from the stored weights.
        :type weights: numpy.ndarray, optional
        :return: the value of the objective function
        :rtype: float
        """
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
            action_old = action_buffer[:, k - 1]
            action_next = action_buffer[:, k]

            # Temporal difference

            critic_old = self.model(observation_old, action_old, weights=weights)
            critic_next = self.model(
                observation_next, action_next, use_stored_weights=True
            )

            temporal_difference = (
                critic_old
                - self.discount_factor * critic_next
                - self.running_objective(observation_old, action_old)
            )

            critic_objective += 1 / 2 * temporal_difference ** 2

        return critic_objective


class CriticCALF(CriticOfObservation):
    def __init__(
        self,
        *args,
        safe_decay_rate=1e-3,
        is_dynamic_decay_rate=True,
        predictor=None,
        observation_init=None,
        safe_controller=None,
        penalty_param=0,
        is_predictive=True,
        **kwargs
    ):
        """
        Initialize a CriticCALF object.

        :param args: Arguments to be passed to the base class `CriticOfObservation`.
        :param safe_decay_rate: Rate at which the safe set shrinks over time.
        :param is_dynamic_decay_rate: Whether the decay rate should be dynamic or not.
        :param predictor: A predictor object to be used to predict future observations.
        :param observation_init: Initial observation to be used to initialize the safe set.
        :param safe_controller: Safe controller object to be used to compute stabilizing actions.
        :param penalty_param: Penalty parameter to be used in the CALF objective.
        :param is_predictive: Whether the safe constraints should be computed based on predictions or not.
        :param kwargs: Keyword arguments to be passed to the base class `CriticOfObservation`.
        """
        super().__init__(*args, **kwargs)
        self.safe_decay_rate = safe_decay_rate
        self.is_dynamic_decay_rate = is_dynamic_decay_rate
        self.safe_controller = safe_controller
        self.predictor = predictor
        self.observation_last_good = observation_init
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
            # self.CALF_decay_constraint = self.CALF_decay_constraint_predicted_on_policy
        else:
            self.CALF_decay_constraint = self.CALF_decay_constraint_no_prediction

        self.intrinsic_constraints = [
            self.CALF_decay_constraint,
        ]

    def update_buffers(self, observation, action):
        """
        Update data buffers and dynamic safe decay rate.

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
        self.update_outcome(observation, action)

        self.current_observation = observation
        self.current_action = action

        if self.is_dynamic_decay_rate:
            print(self.safe_decay_rate)
            self.safe_decay_rate = 1e2 * rc.norm_2(observation)

    def CALF_decay_constraint_no_prediction(self, weights):
        """
        Constraint that ensures that the CALF value is decreasing by a certain rate. The rate is determined by the
        `safe_decay_rate` parameter. This constraint is used when there is no prediction of the next state.

        :param weights: critic weights to be evaluated
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """

        critic_curr = self.model(self.current_observation, weights=weights)
        critic_prev = self.model(self.observation_last_good, use_stored_weights=True)

        self.stabilizing_constraint_violation = (
            critic_curr
            - critic_prev
            + self.predictor.pred_step_size * self.safe_decay_rate
        )
        return self.stabilizing_constraint_violation

    def CALF_critic_lower_bound_constraint(
        self, weights,
    ):
        """
        Constraint that ensures that the value of the critic is above a certain lower bound. The lower bound is determined by
        the `current_observation` and a certain constant.

        :param weights: critic weights to be evaluated
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """
        self.lb_constraint_violation = 1e-4 * rc.norm_2(
            self.current_observation
        ) - self.model(self.current_observation, weights=weights)
        return self.lb_constraint_violation

    def CALF_critic_upper_bound_constraint(self, weights):
        """
        Calculate the constraint violation for the CALF decay constraint when no prediction is made.

        :param weights: critic weights
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """
        self.ub_constraint_violation = self.model(
            self.current_observation, weights=weights
        ) - 1e3 * rc.norm_2(self.current_observation)
        return self.ub_constraint_violation

    def CALF_decay_constraint_predicted_safe_policy(self, weights):
        """
        Calculate the constraint violation for the CALF decay constraint when a predicted safe policy is used.

        :param weights: critic weights
        :type weights: ndarray
        :return: constraint violation
        :rtype: float
        """
        observation_last_good = self.observation_last_good

        self.safe_action = action = self.safe_controller.compute_action(
            self.current_observation
        )
        self.predicted_observation = predicted_observation = self.predictor.predict(
            self.current_observation, action
        )

        self.critic_next = self.model(predicted_observation, weights=weights)
        self.critic_current = self.model(observation_last_good, use_stored_weights=True)

        self.stabilizing_constraint_violation = (
            self.critic_next
            - self.critic_current
            + self.predictor.pred_step_size * self.safe_decay_rate
        )
        return self.stabilizing_constraint_violation

    def CALF_decay_constraint_predicted_on_policy(self, weights):
        """
        Constraint for ensuring that the CALF function decreases at each iteration.
        This constraint is used when prediction is done using the last action taken.

        :param weights: Current weights of the critic network.
        :type weights: ndarray
        :return: Violation of the constraint. A positive value indicates violation.
        :rtype: float
        """
        action = self.action_buffer[:, -1]
        predicted_observation = self.predictor.predict(self.current_observation, action)
        self.stabilizing_constraint_violation = (
            self.model(predicted_observation, weights=weights)
            - self.model(self.observation_last_good, use_stored_weights=True)
            + self.predictor.pred_step_size * self.safe_decay_rate
        )
        return self.stabilizing_constraint_violation

    def objective(self, *args, **kwargs):
        """
        Objective of the critic, which is the sum of the squared temporal difference and the penalty
        for violating the CALF decay constraint, if the penalty parameter is non-zero.

        :param args: Positional arguments to be passed to the parent class's `objective` method.
        :param kwargs: Keyword arguments to be passed to the parent class's `objective` method.
        :return: Value of the objective.
        :rtype: float
        """
        objective = super().objective(*args, **kwargs)

        weights = kwargs.get("weights")
        if self.penalty_param != 0:
            objective += rc.penalty_function(
                self.CALF_decay_constraint(weights), self.penalty_param
            )

        return objective


class CriticTrivial(Critic):
    """
    This is a dummy to calculate outcome (accumulated running objective).

    """

    def __init__(self, running_objective, *args, sampling_time=0.01, **kwargs):
        """
        Initialize a trivial critic.

        :param running_objective: Function object representing the running objective.
        :type running_objective: function
        :param sampling_time: Sampling time.
        :type sampling_time: float
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        self.running_objective = running_objective
        self.sampling_time = sampling_time
        self.outcome = 0
        self.model = ModelWeightContainer(1)
        self.model.weights = None
        self.clock = Clock(sampling_time)

        class optimizer:
            def __init__(self):
                self.engine = None

        self.optimizer = optimizer()
        self.intrinsic_constraints = []
        self.optimized_weights = []

    def __call__(self):
        """
        Returns the current outcome.

        :return: Current outcome.
        :rtype: float
        """
        return self.outcome

    def objective(self, weights):
        """
        Dummy method for the objective function.

        :param weights: Weights.
        :type weights: ndarray or list
        """
        pass

    def get_optimized_weights(self, intrinsic_constraints=None, time=None):
        """
        Dummy method to return optimized weights.

        :param intrinsic_constraints: Constraints to be applied during optimization.
        :type intrinsic_constraints: list of functions
        :param time: Time.
        :type time: float
        :return: Optimized weights.
        :rtype: ndarray or list
        """
        pass

    def update_buffers(self, observation, action):
        """
        Updates the outcome.

        :param observation: Current observation.
        :type observation: ndarray or list
        :param action: Current action.
        :type action: ndarray or list
        """
        self.update_outcome(observation, action)

    def update(self, intrinsic_constraints=None, observation=None, time=None):
        """
        Dummy method for updating the critic.

        :param intrinsic_constraints: Constraints to be applied during optimization.
        :type intrinsic_constraints: list of functions
        :param observation: Current observation.
        :type observation: ndarray or list
        :param time: Time.
        :type time: float
        """
        pass

    def update_outcome(self, observation, action):
        """
        Update the value of the outcome variable by adding the value of the running_objective function
        evaluated at the current observation and action, multiplied by the sampling time.

        :param observation: The current observation.
        :type observation: Any
        :param action: The current action.
        :type action: Any
        """

        self.outcome += self.running_objective(observation, action) * self.sampling_time

    def reset(self):
        """
        Reset the outcome variable to zero.
        """
        self.outcome = 0


class CriticTabularVI(Critic):
    """
    Critic for tabular agents.

    """

    def __init__(
        self,
        dim_state_space,
        running_objective,
        predictor,
        model,
        actor_model,
        discount_factor=1,
        N_parallel_processes=5,
        terminal_state=None,
    ):
        """
        Initialize a CriticTabularVI object.

        :param dim_state_space: The dimensions of the state space.
        :type dim_state_space: tuple of int
        :param running_objective: The running objective function.
        :type running_objective: callable
        :param predictor: The predictor object.
        :type predictor: any
        :param model: The model object.
        :type model: Model
        :param actor_model: The actor model object.
        :type actor_model: any
        :param discount_factor: The discount factor for the temporal difference.
        :type discount_factor: float, optional
        :param N_parallel_processes: The number of parallel processes to use.
        :type N_parallel_processes: int, optional
        :param terminal_state: The terminal state, if applicable.
        :type terminal_state: optional, int or tuple of int
        :return: None
        """

        self.objective_table = rc.zeros(dim_state_space)
        self.action_table = rc.zeros(dim_state_space)
        self.running_objective = running_objective
        self.predictor = predictor
        self.model = model
        self.actor_model = actor_model
        self.discount_factor = discount_factor
        self.N_parallel_processes = N_parallel_processes
        self.terminal_state = terminal_state

    def update_single_cell(self, observation):
        """
        Update the value function for a single state.

        :param observation: current state
        :type observation: tuple of int
        :return: value of the state
        :rtype: float
        """
        action = self.actor_model.weights[observation]
        if tuple(self.terminal_state) == observation:
            return self.running_objective(observation, action)

        return self.objective(observation, action)

    def update(self):
        """
        Update the value function for all states.
        """
        observation_table_indices = tuple(
            [
                (i, j)
                for i in range(self.model.weights.shape[0])
                for j in range(self.model.weights.shape[1])
            ]
        )
        new_table = deepcopy(self.model.weights)
        for observation in observation_table_indices:
            new_table[observation] = self.update_single_cell(observation)
        self.model.weights = new_table

    def objective(self, observation, action):
        """
        Calculate the value of a state given the action taken and the observation of the current state.

        :param observation: current state
        :type observation: tuple of int
        :param action: action taken from the current state
        :type action: int
        :return: value of the state
        :rtype: float
        """
        return (
            self.running_objective(observation, action)
            + self.discount_factor
            * self.model.weights[self.predictor.predict(observation, action)]
        )


class CriticTabularPI(CriticTabularVI):
    def __init__(self, *args, tolerance=1e-3, N_update_iters_max=50, **kwargs):
        """
        Initialize a new instance of the `CriticTabularPI` class.

        :param args: Positional arguments to pass to the superclass's `__init__` method.
        :type args: tuple
        :param tolerance: The tolerance value for the update loop.
        :type tolerance: float
        :param N_update_iters_max: The maximum number of iterations for the update loop.
        :type N_update_iters_max: int
        :param kwargs: Keyword arguments to pass to the superclass's `__init__` method.
        :type kwargs: dict
        """
        super().__init__(*args, **kwargs)
        self.tolerance = tolerance
        self.N_update_iters_max = N_update_iters_max

    def update(self):
        """
        Update the value table.
        """
        observation_table_indices = tuple(
            [
                (i, j)
                for i in range(self.model.weights.shape[0])
                for j in range(self.model.weights.shape[1])
            ]
        )
        for observation in observation_table_indices:
            difference = rc.abs(
                self.model.weights[observation] - self.update_single_cell(observation)
            )
            for _ in range(self.N_update_iters_max):
                new_value = self.update_single_cell(observation)
                difference = self.model.weights[observation] - new_value
                if difference < self.tolerance:
                    self.model.weights[observation] = self.update_single_cell(
                        observation
                    )
                else:
                    break
