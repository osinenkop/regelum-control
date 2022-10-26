"""
This module contains actors, i.e., entities that directly calculate actions.
Actors are inegrated into controllers (agents).

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
from utilities import rc, rej_sampling_rvs
import scipy as sp
from functools import partial


class Actor:
    """
    Class of actors.
    These are to be passed to a `controller`.
    An `objective` (a loss) as well as an `optimizer` are passed to an `actor` externally.
    """

    def reset(self):
        self.action_old = self.action_init
        self.action = self.action_init

    def __init__(
        self,
        prediction_horizon,
        dim_input,
        dim_output,
        control_mode,
        action_bounds=[],
        action_init=[],
        predictor=[],
        optimizer=None,
        critic=[],
        running_objective=[],
        model=None,
        discount_factor=1,
    ):
        self.prediction_horizon = prediction_horizon
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.control_mode = control_mode
        self.action_bounds = action_bounds
        self.optimizer = optimizer
        self.critic = critic
        self.running_objective = running_objective
        self.model = model
        self.predictor = predictor
        self.discount_factor = discount_factor
        self.g_actor_values = []

        if self.control_mode != "MPC" and self.critic == []:
            raise ValueError(
                f"Critic should be passed to actor in {self.control_mode} mode"
            )
        elif self.control_mode == "MPC" and self.running_objective == []:
            raise ValueError(
                f"Stage objective should be passed to actor in {self.control_mode} mode"
            )

        if isinstance(self.action_bounds, (list, np.ndarray)):
            if len(self.action_bounds) > 0:
                self.action_min = np.array(self.action_bounds)[:, 0]
                self.action_max = np.array(self.action_bounds)[:, 1]
        else:
            self.action_min = np.array(self.action_bounds.lb[: self.dim_input])
            self.action_max = np.array(self.action_bounds.ub[: self.dim_input])

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

    def create_observation_constraints(
        self, constraint_functions, action_sequence_reshaped, observation
    ):
        """
        Method to create observation (or state) related constraints using a `predictor` over a `prediction_horizon`.
        In general, we presume that an actor (subsequently, a controller) cannot directly see the state, merely an observation.
        If the output function happens to be an identity though, observation and state can be used interchangeably.
        Notice that constraints are effectively posed on the action (or action sequence).
        The latter is the decision variable for an optimizer that was passed to the actor.
        However, conceptually, the constraints here are related to the observation, whence the naming.
        The end result is passed to an `optimizer`.
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
        )

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

    def update(self, observation, constraint_functions=(), time=None):
        """
        Method to update the current action or weight tensor.
        The old (previous) action or weight tensor is stored.
        The `time` argument is used for debugging purposes.
        """

        # IF NO MODEL IS PASSED, DO ACTION UPDATE. OTHERWISE, WEIGHT UPDATE

        action_sequence_old = rc.rep_mat(
            self.action_old, 1, self.prediction_horizon + 1
        )

        action_sequence_init_reshaped = rc.reshape(
            action_sequence_old, [(self.prediction_horizon + 1) * self.dim_input,],
        )

        constraints = ()

        if self.optimizer.engine == "CasADi":
            action_sequence_init_reshaped = rc.DM(action_sequence_init_reshaped)

            symbolic_dummy = rc.array_symb((1, 1))

            symbolic_var = rc.array_symb(
                tup=rc.shape(action_sequence_init_reshaped), prototype=symbolic_dummy
            )

            actor_objective = lambda action_sequence: self.objective(
                action_sequence, observation
            )

            actor_objective = rc.lambda2symb(actor_objective, symbolic_var)

            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var, observation
                )

            action_sequence_optimized = self.optimizer.optimize(
                actor_objective,
                action_sequence_init_reshaped,
                self.action_bounds,
                constraints=constraints,
                decision_variable_symbolic=symbolic_var,
            )

        elif self.optimizer.engine == "SciPy":
            actor_objective = rc.func_to_lambda_with_params(
                self.objective, observation, var_prototype=action_sequence_init_reshaped
            )

            if constraint_functions:
                constraints = sp.optimize.NonlinearConstraint(
                    partial(
                        self.create_constraints,
                        constraint_functions=constraint_functions,
                        observation=observation,
                    ),
                    -np.inf,
                    0,
                )

            action_sequence_optimized = self.optimizer.optimize(
                actor_objective,
                action_sequence_init_reshaped,
                self.action_bounds,
                constraints=constraints,
            )

        self.model.update_and_cache_weights(action_sequence_optimized[: self.dim_input])
        self.action_old = self.model.cache.weights
        self.action = self.model.weights


class ActorMPC(Actor):
    def objective(
        self, action_sequence, observation,
    ):
        """
        Model-predictive control (MPC) actor.
        Optimizes the following actor objective:
        .. math::
            J^a \left( y_k| \{u\}_k^{N_a+1} \right) = \sum_{i=0}^{N_a} \gamma^i r(y_{i|k}, u_{i|k})

        Notation:

        * :math:`y`: observation
        * :math:`u`: action
        * :math:`N_a`: prediction horizon
        * :math:`gamma`: discount factor
        * :math:`r`: running objective function
        * :math:`\{\bullet\}_k^N`: sequence from index :math:`k` to index :math:`k+N-1`
        * :math:`\bullet_{i|k}`: element in a sequence with index :math:`k+i-1`
        """
        action_sequence_reshaped = rc.reshape(
            action_sequence, [self.prediction_horizon + 1, self.dim_input]
        )

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_cur = rc.reshape(observation, [1, self.dim_output])

        observation_sequence = rc.vstack(
            (observation_cur, observation_sequence_predicted)
        )

        actor_objective = 0
        for k in range(self.prediction_horizon):
            actor_objective += self.discount_factor ** k * self.running_objective(
                observation_sequence[k, :].T, action_sequence_reshaped[k, :].T
            )
        return actor_objective


class ActorSQL(Actor):
    def objective(
        self, action_sequence, observation,
    ):
        """
        Staked Q-learning (SQL) actor.
        Optimizes the following actor objective:
        .. math::
            J^a \left( y_k| \{u\}_k^{N_a+1} \right) = \sum_{i=0}^{N_a} \gamma^i Q(y_{i|k}, u_{i|k})

        Notation:

        * :math:`y`: observation
        * :math:`u`: action
        * :math:`N_a`: prediction horizon
        * :math:`gamma`: discount factor
        * :math:`r`: running objective function
        * :math:`Q`: action-objective function (or its estimate)
        * :math:`\{\bullet\}_k^N`: sequence from index :math:`k` to index :math:`k+N-1`
        * :math:`\bullet_{i|k}`: element in a sequence with index :math:`k+i-1`
        """

        action_sequence_reshaped = rc.reshape(
            action_sequence, [self.prediction_horizon + 1, self.dim_input]
        )

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_sequence = rc.vstack(
            (
                rc.reshape(observation, [1, self.dim_output]),
                observation_sequence_predicted,
            )
        )

        actor_objective = 0

        for k in range(self.prediction_horizon + 1):
            action_objective = self.critic(
                observation_sequence[k, :],
                action_sequence_reshaped[k, :],
                use_stored_weights=True,
            )

            actor_objective += action_objective
        return actor_objective


class ActorRQL(Actor):
    def objective(
        self, action_sequence, observation,
    ):
        """
        Rollout Q-learning (RQL) actor.
        Optimizes the following actor objective:
        .. math::
            J^a \left( y_k| \{u\}_k^{N_a+1} \right) = \sum_{i=0}^{N_a-1} \gamma^i r(y_{i|k}, u_{i|k}) + \gamma^{N_a} Q(y_{N_a|k}, u_{N_a|k})

        Notation:

        * :math:`y`: observation
        * :math:`u`: action
        * :math:`N_a`: prediction horizon
        * :math:`gamma`: discount factor
        * :math:`r`: running objective function
        * :math:`Q`: action-objective function (or its estimate)
        * :math:`\{\bullet\}_k^N`: sequence from index :math:`k` to index :math:`k+N-1`
        * :math:`\bullet_{i|k}`: element in a sequence with index :math:`k+i-1`
        """

        action_sequence_reshaped = rc.reshape(
            action_sequence, [self.prediction_horizon + 1, self.dim_input]
        )

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_sequence = rc.vstack(
            (
                rc.reshape(observation, [1, self.dim_output]),
                observation_sequence_predicted,
            )
        )

        actor_objective = 0

        for k in range(self.prediction_horizon):
            actor_objective -= self.discount_factor ** k * self.running_objective(
                observation_sequence[k, :], action_sequence_reshaped[k, :]
            )

        # actor_objective += (
        #     self.discount_factor ** self.prediction_horizon
        #     * self.critic(
        #         observation_sequence[-1, :],
        #         action_sequence_reshaped[-1, :],
        #         use_stored_weights=True,
        #     )
        # )
        return actor_objective


class ActorV(Actor):
    def objective(
        self, action, observation,
    ):
        """
        Actor minimizing the sum of the running objective and the optimal (or estimate thereof) objective of the next step.
        May be suitable for value iteration and policy iteration agents.
        Specifically, it optimizes the following actor objective:
        .. math::
            J^a \left( y_k| \{u\}_k \right) =  r(y_{k}, u_{k}) + \gamma J^*(y_{k})

        Notation:

        * :math:`y`: observation
        * :math:`u`: action
        * :math:`gamma`: discount factor
        * :math:`r`: running objective function
        * :math:`J^*`: optimal objective function (or its estimate)
        """

        action_sequence_reshaped = rc.reshape(action, [1, self.dim_input])

        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation, action_sequence_reshaped
        )

        observation_sequence = rc.vstack(
            (
                rc.reshape(observation, [1, self.dim_output]),
                observation_sequence_predicted,
            )
        )

        actor_objective = self.running_objective(
            observation_sequence[0, :], action_sequence_reshaped
        ) + self.gamma * self.critic(
            observation_sequence[1, :], use_stored_weights=True
        )

        return actor_objective


class ActorSTAG(ActorV):
    """
    Stabilizing agent (STAG) actor.
    It finds actions subject to specially designed stabilizing constraints, which involve a backup stabilizing policy.
    """

    def __init__(self, *args, eps=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def get_optimized_action(self, observation, constraint_functions=(), time=None):

        action_sequence_old = rc.rep_mat(
            self.action_old, 1, self.prediction_horizon + 1
        )

        action_sequence_init_reshaped = rc.reshape(
            action_sequence_old, [(self.prediction_horizon + 1) * self.dim_input],
        )

        constraints = ()

        def stailizing_constraint(action, observation):

            observation_next = self.predictor.predict_state(observation, action)

            critic_curr = self.critic(observation)
            critic_next = self.critic(observation_next)

            return (
                critic_next
                - critic_curr
                + self.predictor.pred_step_size * self.critic.safe_decay_rate
            )

        if self.optimizer.engine == "CasADi":
            actor_objective, action_sequence_symbolic = rc.func_to_lambda_with_params(
                self.objective, observation, var_prototype=action_sequence_init_reshaped
            )

            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, action_sequence_symbolic, observation
                )

            lambda_constraint = (
                lambda action: stailizing_constraint(action, observation) - self.eps
            )

            constraints += (
                rc.lambda2symb(lambda_constraint, action_sequence_symbolic),
            )

            action_sequence_optimized = self.optimizer.optimize(
                actor_objective,
                action_sequence_init_reshaped,
                self.action_bounds,
                constraints=constraints,
                decision_variable_symbolic=action_sequence_symbolic,
            )

        elif self.optimizer.engine == "SciPy":
            actor_objective = rc.func_to_lambda_with_params(
                self.objective, observation, var_prototype=action_sequence_init_reshaped
            )

            if constraint_functions:
                constraints = sp.optimize.NonlinearConstraint(
                    partial(
                        self.create_scipy_constraints,
                        constraint_functions=constraint_functions,
                        observation=observation,
                    ),
                    -np.inf,
                    0,
                )
            resulting_constraints = sp.optimize.NonlinearConstraint(
                lambda action: stailizing_constraint(action, observation),
                -np.inf,
                self.eps,
            )

            action_sequence_optimized = self.optimizer.optimize(
                actor_objective,
                action_sequence_init_reshaped,
                self.action_bounds,
                constraints=resulting_constraints,
            )

        return action_sequence_optimized[: self.dim_input]


class ActorTabular(ActorV):
    """
    Actor minimizing the sum of the running objective and the optimal (or estimate thereof) objective of the next step.
    May be suitable for value iteration and policy iteration agents.
    Specifically, it optimizes the following actor objective:
    .. math::
        J^a \left( y_k| \{u\}_k \right) =  r(y_{k}, u_{k}) + \gamma J^*(y_{k})

    Notation:

    * :math:`y`: observation
    * :math:`u`: action
    * :math:`gamma`: discount factor
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

        new_action_table = self.optimizer.optimize(self.objective, self.model.weights)

        self.model.update_and_cache_weights(new_action_table)

    def objective(
        self, action, observation,
    ):
        if tuple(observation) == tuple(self.terminal_state):
            return 0

        observation_predicted = self.predictor.predict_sequence(observation, action)

        actor_objective = self.running_objective(
            observation, action
        ) + self.discount_factor * self.critic(observation_predicted)

        return actor_objective


class ActorProbabilisticEpisodic(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients = []

    def update(self, observation):
        action_sample = self.model.sample_from_distribution(observation)
        self.action = np.array(
            np.clip(action_sample, self.action_bounds[0], self.action_bounds[1])
        )
        self.action_old = self.action
        current_gradient = self.model.compute_gradient(action_sample)
        self.store_gradient(current_gradient)

    def reset(self):
        super().reset()
        self.gradients = []

    def update_weights_by_gradient(self, gradient, learning_rate):
        model_weights = self.model.weights
        new_model_weights = rc.array(
            model_weights - learning_rate * gradient * rc.array([1, 0.0, 1])
        )

        self.model.update(new_model_weights)

    def store_gradient(self, gradient):
        self.gradients.append(gradient)

    def get_action(self):
        return self.action


class ActorProbabilisticEpisodicAC(ActorProbabilisticEpisodic):
    def update(self, observation):
        action_sample = self.model.sample_from_distribution(observation)
        self.action = np.array(
            np.clip(action_sample, self.action_bounds[0], self.action_bounds[1])
        )
        self.action_old = self.action

        Q_value = self.critic(observation, action_sample).detach().numpy()
        current_gradient = self.model.compute_gradient(action_sample) * Q_value

        self.store_gradient(current_gradient)
