"""
This module contains actors, i.e., entities that directly calculate actions.
Actors are inegrated into controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""


import numpy as np
import scipy as sp
from functools import partial
from abc import ABC, abstractmethod
from typing import Union

from .__utilities import rc
from .callbacks import introduce_callbacks, apply_callbacks
from .predictors import Predictor
from .optimizers import Optimizer
from .critics import Critic
from .models import Model


class Actor:
    """
    Class of actors.
    These are to be passed to a `controller`.
    An `objective` (a loss) as well as an `optimizer` are passed to an `actor` externally.
    """

    def __call__(self, observation):
        """
        Return the most recent action taken by the actor.
        
        :param observation: Current observation of the system.
        :type observation: ndarray
        :returns: Most recent action taken by the actor.
        :rtype: ndarray
        """
        return self.action

    def reset(self):
        """
        Reset the actor to its initial state.
        """
        self.action_old = self.action_init
        self.action = self.action_init

    def __init__(
        self,
        dim_output: int = 5,
        dim_input: int = 2,
        prediction_horizon: int = 1,
        action_bounds: Union[list, np.ndarray] = None,
        action_init: list = None,
        predictor: Predictor = None,
        optimizer: Optimizer = None,
        critic: Critic = None,
        running_objective=None,
        model: Model = None,
        discount_factor=1,
    ):
        """
        Initialize an actor.
        
        :param prediction_horizon: Number of time steps to look into the future.
        :type prediction_horizon: int
        :param dim_input: Dimension of the observation.
        :type dim_input: int
        :param dim_output: Dimension of the action.
        :type dim_output: int
        :param action_bounds: Bounds on the action.
        :type action_bounds: list or ndarray, optional
        :param action_init: Initial action.
        :type action_init: list, optional
        :param predictor: Predictor object for generating predictions.
        :type predictor: Predictor, optional
        :param optimizer: Optimizer object for optimizing the action.
        :type optimizer: Optimizer, optional
        :param critic: Critic object for evaluating actions.
        :type critic: Critic, optional
        :param running_objective: Running objective object for recording the running objective.
        :type running_objective: RunningObjective, optional
        :param model: Model object to be used as reference by the Predictor and the Critic.
        :type model: Model, optional
        :param discount_factor: discount factor to be used in conjunction with the critic.
        :type discount_factor: float, optional
        """
        self.prediction_horizon = prediction_horizon
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.action_bounds = action_bounds
        self.optimizer = optimizer
        self.critic = critic
        self.running_objective = running_objective
        self.model = model
        self.predictor = predictor
        self.discount_factor = discount_factor

        if isinstance(self.action_bounds, (list, np.ndarray)):
            if len(self.action_bounds) > 0:
                self.action_min = np.array(self.action_bounds)[:, 0]
                self.action_max = np.array(self.action_bounds)[:, 1]
        else:
            self.action_min = np.array(self.action_bounds.lb[: self.dim_output])
            self.action_max = np.array(self.action_bounds.ub[: self.dim_output])

        if len(action_init) == 0:
            self.action_old = (self.action_min + self.action_max) / 2
            self.action_sequence_init = rc.rep_mat(
                self.action_old, 1, self.prediction_horizon + 1
            )
            self.action_init = self.action_old
        else:
            self.action_init = action_init
            self.action_old = action_init
            self.action_sequence_init = rc.rep_mat(
                action_init, 1, self.prediction_horizon + 1
            )

        self.action_sequence_min = rc.rep_mat(
            self.action_min, 1, prediction_horizon + 1
        )
        self.action_sequence_max = rc.rep_mat(
            self.action_max, 1, prediction_horizon + 1
        )
        self.action_bounds = [self.action_sequence_min, self.action_sequence_max]
        self.action = self.action_old
        self.intrinsic_constraints = []

    def create_observation_constraints(
        self, constraint_functions, action_sequence_reshaped, observation
    ):
        """
        Method to create observation (or state) related constraints using a `predictor` over a `prediction_horizon`.
        These constraints are related to observations, not actions, although they are ultimately imposed on the action
        (or action sequence), which is the decision variable for the `optimizer` passed to the `actor`.
        The `predictor` is used to generate a sequence of predicted observations over the `prediction_horizon`.
        Constraint functions are then applied to each element of the predicted observation sequence.
        The maximum constraint violation over the entire sequence is returned and passed to the `optimizer`.

        :param constraint_functions: List of functions that take an observation as input and return a scalar constraint
                                    violation.
        :type constraint_functions: list[function]
        :param action_sequence_reshaped: Flat action sequence array with shape (prediction_horizon * dim_input).
        :type action_sequence_reshaped: numpy.ndarray
        :param observation: Current observation of the system.
        :type observation: numpy.ndarray
        :return: Maximum constraint violation over the entire predicted observation sequence.
        :rtype: float
        """
        current_observation = observation

        resulting_constraints = [0 for _ in range(self.prediction_horizon - 1)]
        constraint_violation_buffer = [0 for _ in constraint_functions]

        for constraint_function in constraint_functions:
            constraint_violation_buffer[0] = constraint_function(current_observation)

        max_constraint_violation = rc.max(constraint_violation_buffer)

        max_constraint_violation = -1
        action_sequence = rc.reshape(
            action_sequence_reshaped, [self.prediction_horizon, self.dim_input]
        ).T

        # Initialize for caclulation of the predicted observation sequence
        predicted_observation = current_observation

        for i in range(1, self.prediction_horizon):

            current_action = action_sequence[i - 1, :]
            current_state = predicted_observation

            predicted_observation = self.predictor.predict_state(
                current_state, current_action
            )

            constraint_violation_buffer = []
            for constraint in constraint_functions:
                constraint_violation_buffer.append(constraint(predicted_observation))

            max_constraint_violation = rc.max(constraint_violation_buffer)
            resulting_constraints[i - 1] = max_constraint_violation

        for i in range(2, self.prediction_horizon - 1):
            resulting_constraints[i] = rc.if_else(
                resulting_constraints[i - 1] > 0,
                resulting_constraints[i - 1],
                resulting_constraints[i],
            )

        return resulting_constraints

    def receive_observation(self, observation):
        """
        Update the current observation of the actor.
        :param observation: The current observation.
        :type observation: numpy array
        """
        self.observation = observation

    def set_action(self, action):
        """
        Set the current action of the actor.
        :param action: The current action.
        :type action: numpy array
        """
        self.action_old = self.action
        self.action = action

    def update_action(self, observation=None):
        """
        Update the current action of the actor.
        :param observation: The current observation. If not provided, the previously received observation will be used.
        :type observation: numpy array, optional
        """
        self.action_old = self.action

        if observation is None:
            observation = self.observation

        self.action = self.model(observation)

    def update_weights(self, weights=None):
        """
        Update the weights of the model of the actor.
        :param weights: The weights to update the model with. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        if weights is None:
            self.model.update_weights(self.optimized_weights)
        else:
            self.model.update_weights(weights)

    def cache_weights(self, weights=None):
        """
        Cache the current weights of the model of the actor.
        :param weights: The weights to cache. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        if weights is not None:
            self.model.cache_weights(weights)
        else:
            self.model.cache_weights(self.optimized_weights)

    def update_and_cache_weights(self, weights=None):
        """
        Update and cache the weights of the model of the actor.
        :param weights: The weights to update and cache. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        self.update_weights(weights)
        self.cache_weights(weights)

    def restore_weights(self):
        """
        Restore the previously cached weights of the model of the actor.
        """
        self.model.restore_weights()
        self.set_action(self.action_old)

    def accept_or_reject_weights(
        self, weights, constraint_functions=None, optimizer_engine="SciPy", atol=1e-10
    ):
        """
        Determines whether the given weights should be accepted or rejected based on the specified constraints.
        
        :param weights: Array of weights to be evaluated.
        :type weights: np.ndarray
        :param constraint_functions: List of constraint functions to be evaluated.
        :type constraint_functions: Optional[List[Callable[[np.ndarray], float]]], optional
        :param optimizer_engine: String indicating the optimization engine being used.
        :type optimizer_engine: str, optional
        :param atol: Absolute tolerance used when evaluating the constraints.
        :type atol: float, optional
        :return: String indicating whether the weights were accepted ("accepted") or rejected ("rejected").
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

    def optimize_weights(self, constraint_functions=None, time=None):
        """
        Method to optimize the current actor weights. The old (previous) weights are stored.
        The `time` argument is used for debugging purposes.
        If weights satisfying constraints are found, the method returns the status `accepted`.
        Otherwise, it returns the status `rejected`.

        :param constraint_functions: List of functions defining constraints on the optimization.
        :type constraint_functions: list of callables, optional
        :param time: Debugging parameter to track time during optimization process.
        :type time: float, optional
        :returns: String indicating whether the optimization process was accepted or rejected.
        :rtype: str
        """
        final_count_of_actions = self.prediction_horizon + 1
        action_sequence = rc.rep_mat(self.action, 1, final_count_of_actions)

        action_sequence_init_reshaped = rc.reshape(
            action_sequence, [final_count_of_actions * self.dim_output],
        )

        constraints = []

        if self.optimizer.engine == "CasADi":
            action_sequence_init_reshaped = rc.DM(action_sequence_init_reshaped)

            symbolic_dummy = rc.array_symb((1, 1))

            symbolic_var = rc.array_symb(
                tup=rc.shape(action_sequence_init_reshaped), prototype=symbolic_dummy
            )

            actor_objective = lambda action_sequence: self.objective(
                action_sequence, self.observation
            )

            actor_objective = rc.lambda2symb(actor_objective, symbolic_var)
            constraint_functions = []
            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var, self.observation
                )

            if self.intrinsic_constraints:
                intrisic_constraints = [
                    rc.lambda2symb(constraint, symbolic_var)
                    for constraint in self.intrinsic_constraints
                ]
            else:
                intrisic_constraints = []

            self.optimized_weights = self.optimizer.optimize(
                actor_objective,
                action_sequence_init_reshaped,
                self.action_bounds,
                constraints=intrisic_constraints + constraint_functions,
                decision_variable_symbolic=symbolic_var,
            )
            # self.cost_function = actor_objective
            # self.constraint = intrisic_constraints[0]
            # self.weights_init = action_sequence_init_reshaped
            # self.symbolic_var = symbolic_var

        elif self.optimizer.engine == "SciPy":
            actor_objective = rc.function_to_lambda_with_params(
                self.objective, self.observation,
            )

            if constraint_functions is not None:
                constraints = sp.optimize.NonlinearConstraint(
                    partial(
                        self.create_constraints,
                        constraint_functions=constraint_functions,
                        observation=self.observation,
                    ),
                    -np.inf,
                    0,
                )
            if self.intrinsic_constraints:
                intrinsic_constraints = [
                    sp.optimize.NonlinearConstraint(constraint_function, -np.inf, 0,)
                    for constraint_function in self.intrinsic_constraints
                ]
            else:
                intrinsic_constraints = []

            self.optimized_weights = self.optimizer.optimize(
                actor_objective,
                action_sequence_init_reshaped,
                self.action_bounds,
                constraints=constraints + intrinsic_constraints,
            )

        if self.intrinsic_constraints:
            # DEBUG ==============================
            # print("with constraint functions")
            # /DEBUG =============================
            self.weights_acceptance_status = self.accept_or_reject_weights(
                self.optimized_weights,
                constraint_functions=self.intrinsic_constraints,
                optimizer_engine=self.optimizer.engine,
            )
        else:
            # DEBUG ==============================
            # print("without constraint functions")
            # /DEBUG =============================
            self.weights_acceptance_status = "accepted"

        return self.weights_acceptance_status


class ActorMPC(Actor):
    """
    Model-predictive control (MPC) actor.
    Optimizes the following actor objective:
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
        self, action_sequence, observation,
    ):
        """
        Calculates the actor objective for the given action sequence and observation using Model Predictive Control (MPC).

        :param action_sequence: sequence of actions to be evaluated in the objective function
        :type action_sequence: numpy.ndarray
        :param observation: current observation
        :type observation: numpy.ndarray
        :return: the actor objective for the given action sequence
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

        actor_objective = 0
        for k in range(self.prediction_horizon + 1):
            actor_objective += self.discount_factor ** k * self.running_objective(
                observation_sequence[:, k], action_sequence_reshaped[:, k]
            )
        return actor_objective


class ActorSQL(Actor):
    """
    Staked Q-learning (SQL) actor.
    Optimizes the following actor objective:
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
        self, action_sequence, observation,
    ):
        """
        Calculates the actor objective for the given action sequence and observation using the stacked Q-learning (SQL) algorithm.
        
        :param action_sequence: numpy array of shape (prediction_horizon+1, dim_output) representing the sequence of actions to optimize
        :type action_sequence: numpy.ndarray
        :param observation: numpy array of shape (dim_output,) representing the current observation
        :type observation: numpy.ndarray
        :return: actor objective for the given action sequence and observation
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
            (observation, observation_sequence_predicted,)
        )

        actor_objective = 0

        for k in range(self.prediction_horizon + 1):
            action_objective = self.critic(
                observation_sequence[:, k],
                action_sequence_reshaped[:, k],
                use_stored_weights=True,
            )

            actor_objective += action_objective
        return actor_objective


class ActorRQL(Actor):
    """
    Rollout Q-learning (RQL) actor.
    Optimizes the following actor objective:

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
        self, action_sequence, observation,
    ):
        """
        Calculates the actor objective for the given action sequence and observation using Rollout Q-learning (RQL).
        
        :param action_sequence: numpy array of shape (prediction_horizon+1, dim_output) representing the sequence of actions to optimize
        :type action_sequence: numpy.ndarray
        :param observation: numpy array of shape (dim_output,) representing the current observation
        :type observation: numpy.ndarray
        :return: actor objective for the given action sequence and observation
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
            (observation, observation_sequence_predicted,)
        )

        actor_objective = 0

        for k in range(self.prediction_horizon):
            actor_objective += self.discount_factor ** k * self.running_objective(
                observation_sequence[:, k], action_sequence_reshaped[:, k]
            )

        actor_objective += self.critic(
            action_sequence_reshaped[:, -1], observation_sequence[:, -1]
        )

        return actor_objective


class ActorRPO(Actor):
    """
    Running (objective) Plus Optimal (objective) actor.
    Actor minimizing the sum of the running objective and the optimal (or estimate thereof) objective of the next step.
    May be suitable for value iteration and policy iteration agents.
    Specifically, it optimizes the following actor objective:

    :math:`J^a \\left( y_k| \\{u\\}_k \\right) =  r(y_{k}, u_{k}) + \\gamma J^*(y_{k})`

    Notation:

    * :math:`y`: observation
    * :math:`u`: action
    * :math:`\\gamma`: discount factor
    * :math:`r`: running objective function
    * :math:`J^*`: optimal objective function (or its estimate)
    """

    def objective(
        self, action_sequence, observation,
    ):
        """
        Calculates the actor objective for the given action sequence and observation using Running Plus Optimal (RPO).
        
        :param action_sequence: numpy array of shape (prediction_horizon+1, dim_input) representing the sequence of actions to optimize
        :type action_sequence: numpy.ndarray
        :param observation: numpy array of shape (dim_input,) representing the current observation
        :type observation: numpy.ndarray
        :return: actor objective for the given action sequence and observation
        :rtype: float
        """
        current_action = action_sequence[: self.dim_output]

        observation_predicted = self.predictor.predict(observation, current_action)

        running_objective_value = self.running_objective(observation, current_action)

        critic_of_observation = self.critic(observation_predicted)

        actor_objective = running_objective_value + critic_of_observation

        return actor_objective


class ActorCALF(ActorRPO):
    """Actor using Critic As a Lyapunov Function (CALF) to constrain the optimization."""

    def __init__(
        self,
        safe_controller,
        *args,
        actor_constraints_on=True,
        penalty_param=0,
        actor_regularization_param=0,
        **kwargs,
    ):
        """
        Initialize the actor with a safe controller, and optional arguments for constraint handling, penalty term, and actor regularization.

        :param safe_controller: controller used to compute a safe action in case the optimization is rejected
        :type safe_controller: Controller
        :param actor_constraints_on: whether to use the CALF constraints in the optimization
        :type actor_constraints_on: bool
        :param penalty_param: penalty term for the optimization objective
        :type penalty_param: float
        :param actor_regularization_param: regularization term for the actor weights
        :type actor_regularization_param: float
        """
        super().__init__(*args, **kwargs)
        self.safe_controller = safe_controller
        self.penalty_param = penalty_param
        self.actor_regularization_param = actor_regularization_param
        self.predictive_constraint_violations = []
        self.intrinsic_constraints = (
            [
                self.CALF_decay_constraint_for_actor,
                # self.CALF_decay_constraint_for_actor_same_critic
            ]
            if actor_constraints_on
            else []
        )
        self.weights_acceptance_status = False
        safe_action = self.safe_controller.compute_action(
            self.critic.observation_last_good
        )
        self.action = safe_action
        self.model.update_and_cache_weights(safe_action)

    def CALF_decay_constraint_for_actor(self, weights):
        """
        Constraint for the actor optimization, ensuring that the critic value will not decrease by less than the required decay rate.

        :param weights: actor weights to be evaluated
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
            + self.critic.sampling_time * self.critic.safe_decay_rate
        )
        return self.predictive_constraint_violation

    def CALF_decay_constraint_for_actor_same_critic(self, weights):
        """
        Calculate the predictive constraint violation for the CALF.

        This function calculates the violation of the "CALF decay constraint" which is used to ensure that the critic's value function
        (as a Lyapunov function) decreases over time. This helps to guarantee that the system remains stable.

        :param weights: (array) Weights for the actor model.
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
            + self.critic.sampling_time * self.critic.safe_decay_rate
        )

        return self.predictive_constraint_violation


class ActorCLF(ActorCALF):
    """
    ActorCLF is an actor class that aims to optimize the decay of a Control-Lyapunov function (CLF).
    """

    def __init__(self, *args, **kwargs):
        """
        :param safe_controller: object of class SafeController that provides a safe action if the current action would violate the safe set.
        :type safe_controller: SafeController
        """
        super().__init__(*args, **kwargs)
        self.intrinsic_constraints = []

    def objective(
        self, action, observation,
    ):
        """
        Computes the anticipated decay of the CLF.
        
        :param action: Action taken by the actor.
        :type action: ndarray
        :param observation: Observation of the system.
        :type observation: ndarray
        :return: Actor objective
        :rtype: float
        """

        observation_predicted = self.predictor.predict(observation, action)

        actor_objective = 0

        actor_objective += self.safe_controller.compute_LF(observation_predicted)

        return actor_objective


class ActorTabular(ActorRPO):
    """
    Actor minimizing the sum of the running objective and the optimal (or estimate thereof) objective of the next step.
    May be suitable for value iteration and policy iteration agents.
    Specifically, it optimizes the following actor objective:

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
        """
        Initializes an actorTabular object.

        :param dim_world: The dimensions of the world (i.e. the dimensions of the state space).
        :type dim_world: int
        :param predictor: An object that predicts the next state given an action and the current state.
        :type predictor: object
        :param optimizer: An object that optimizes the actor's objective function.
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
        """
        Updates the action table using the optimizer.
        """

        new_action_table = self.optimizer.optimize(self.objective, self.model.weights)

        self.model.update_and_cache_weights(new_action_table)

    def objective(
        self, action, observation,
    ):
        """
        Calculates the actor objective for a given action and observation.
        The actor objective is defined as the sum of the running objective and the optimal (or estimate thereof) objective of the next step.

        :param action: The action for which the actor objective is to be calculated.
        :type action: np.ndarray
        :param observation: The observation for which the actor objective is to be calculated.
        :type observation: np.ndarray
        :return: The actor objective.
        :rtype: float
        """
        if tuple(observation) == tuple(self.terminal_state):
            return 0

        observation_predicted = self.predictor.predict_sequence(observation, action)

        actor_objective = self.running_objective(
            observation, action
        ) + self.discount_factor * self.critic(observation_predicted)

        return actor_objective


class ActorProbabilisticEpisodic(Actor):
    def __init__(
        self,
        dim_output: int = 5,
        dim_input: int = 2,
        action_bounds=None,
        action_init=None,
        model=None,
        **kwargs,
    ):
        """
        Initialize an actor that samples actions from a probabilistic model.
        The actor also stores gradients for the model weights for each action taken.

        :param action_bounds: Bounds on the action.
        :type action_bounds: list or ndarray, optional
        :param action_init: Initial action.
        :type action_init: list, optional
        :param model: Model object to be used as reference by the Predictor and the Critic.
        :type model: Model, optional
        """
        super().__init__(
            dim_output=dim_output,
            dim_input=dim_input,
            action_bounds=action_bounds,
            action_init=action_init,
            model=model,
            **kwargs,
        )
        self.gradients = []

    def update_action(self, observation):
        """
        Sample an action from the probabilistic model, clip it to the action bounds, and store its gradient.

        :param observation: The current observation.
        """
        action_sample = self.model.sample_from_distribution(observation)
        self.action = np.array(
            np.clip(action_sample, self.action_bounds[0], self.action_bounds[1])
        )
        self.action_old = self.action
        current_gradient = self.model.compute_gradient(action_sample)
        self.store_gradient(current_gradient)

    def reset(self):
        """Reset the actor's stored gradients and call the base `Actor` class's reset method."""
        super().reset()
        self.gradients = []

    def update_weights_by_gradient(self, gradient, learning_rate):
        """
        Update the model weights by subtracting the gradient multiplied by the learning rate and a constant factor.

        :param gradient: The gradient of the model's weights.
        :type gradient: numpy array
        :param learning_rate: The learning rate of the update.
        :type learning_rate: float
        """
        model_weights = self.model.weights
        new_model_weights = rc.array(
            model_weights - learning_rate * gradient * rc.array([1, 0.0, 1])
        )

        self.model.update(new_model_weights)

    def store_gradient(self, gradient):
        """
        Store the gradient of the model's weights.

        :param gradient: The gradient of the model's weights.
        :type gradient: numpy array
        """
        self.gradients.append(gradient)

    def get_action(self):
        """
        Get the current action.

        :return: The current action.
        :rtype: numpy array
        """
        return self.action

    def optimize_weights(self):
        pass

    def update_and_cache_weights(self):
        pass


class ActorProbabilisticEpisodicAC(ActorProbabilisticEpisodic):
    def update(self, observation):
        """
        Samples an action from the actor's distribution, updates the action and action_old attributes,
        and stores the current gradient in the gradients list.
        
        :param observation: The current observation of the environment.
        """
        action_sample = self.model.sample_from_distribution(observation)
        self.action = np.array(
            np.clip(action_sample, self.action_bounds[0], self.action_bounds[1])
        )
        self.action_old = self.action

        Q_value = self.critic(observation, action_sample).detach().numpy()
        current_gradient = self.model.compute_gradient(action_sample) * Q_value

        self.store_gradient(current_gradient)
