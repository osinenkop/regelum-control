"""
Module containing critics, which are integrated in controllers (agents).

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
from .utilities import rc, NUMPY, CASADI, TORCH
from abc import ABC, abstractmethod
import scipy as sp
from functools import partial
import torch
from copy import deepcopy
from multiprocessing import Pool
from .models import ModelWeightContainer


class Critic(ABC):
    """
    Blueprint of a critic.
    """

    def __init__(
        self,
        dim_input,
        dim_output,
        data_buffer_size,
        optimizer=None,
        model=None,
        running_objective=[],
        discount_factor=1,
        observation_target=[],
        sampling_time=None,
    ):

        self.data_buffer_size = data_buffer_size
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.optimizer = optimizer
        if self.optimizer.engine == "Torch":
            self.optimizer_engine = TORCH
        elif self.optimizer.engine == "CasADi":
            self.optimizer_engine = CASADI
        else:
            self.optimizer_engine = NUMPY
        self.initialize_buffers()
        self.observation_target = observation_target

        self.discount_factor = discount_factor
        self.running_objective = running_objective

        # DEBUG-----------------------------------------------------------
        # self.g_critic_values = []
        # /DEBUG----------------------------------------------------------

        self.model = model
        self.current_critic_loss = 0
        self.outcome = 0
        self.sampling_time = sampling_time

    def __call__(self, *args, use_stored_weights=False):
        if len(args) == 2:
            chi = rc.concatenate(tuple(args))
        else:
            chi = args
        return self.model(chi, use_stored_weights=use_stored_weights)

    def accept_or_reject_weights(
        self,
        weights,
        observation=None,
        constraint_functions=None,
        optimizer_engine="SciPy",
    ):

        if constraint_functions is None:
            constraints_not_violated = True
        else:
            constraints_not_violated = all(
                [cond(weights) for cond in constraint_functions]
            )

        if constraints_not_violated:
            if optimizer_engine in ["SciPy", "CasADi"]:
                self.model.update_and_cache_weights(weights)
            elif optimizer_engine == "Torch":
                self.model.update_and_cache_weights()
            return "accepted"
        else:
            if optimizer_engine == "Torch":
                self.model.restore_weights()
            return "rejected"

    def update(
        self, constraint_functions=[], observation=None, time=None,
    ):

        """
        Update of critic weights.

        """
        if self.optimizer.engine == "CasADi":
            optimized_weights = self._CasADi_update(constraint_functions)

        elif self.optimizer.engine == "SciPy":
            optimized_weights = self._SciPy_update(constraint_functions)

        elif self.optimizer.engine == "Torch":
            self._Torch_update()
            optimized_weights = self.model.weights

        if constraint_functions:
            weights_acceptance_status = self.accept_or_reject_weights(
                optimized_weights,
                observation=observation,
                constraint_functions=constraint_functions,
                optimizer_engine=self.optimizer.engine,
            )
        else:
            weights_acceptance_status = True

        return weights_acceptance_status

    def update_buffers(self, observation, action):
        self.action_buffer = rc.push_vec(
            self.action_buffer, rc.array(action, prototype=self.action_buffer)
        )
        self.observation_buffer = rc.push_vec(
            self.observation_buffer,
            rc.array(observation, prototype=self.observation_buffer),
        )
        self.update_outcome(observation, action)

    def initialize_buffers(self):

        self.action_buffer = rc.zeros(
            (int(self.data_buffer_size), int(self.dim_input)),
            rc_type=self.optimizer_engine,
        )
        self.observation_buffer = rc.zeros(
            (int(self.data_buffer_size), int(self.dim_output)),
            rc_type=self.optimizer_engine,
        )

    def update_outcome(self, observation, action):

        self.outcome += self.running_objective(observation, action) * self.sampling_time

    def reset(self):
        self.outcome = 0
        self.current_critic_loss = 0
        self.initialize_buffers()

    # def grad_observation(self, observation):

    #     observation_symbolic = rc.array_symb(rc.shape(observation), literal="x")
    #     weights_symbolic = rc.array_symb(rc.shape(self.weights), literal="w")

    #     critic_func = self.forward(weights_symbolic, observation_symbolic)

    #     f = Function("f", [observation_symbolic, weights_symbolic], [critic_func])

    #     gradient = rc.autograd(f, observation_symbolic, weights_symbolic)

    #     gradient_evaluated = gradient(observation, weights_symbolic)

    #     # Lie_derivative = rc.dot(v, gradient(observation_symbolic))
    #     return gradient_evaluated, weights_symbolic

    def _SciPy_update(self, constraint_functions=[]):

        weights_init = self.model.weight_min

        constraints = ()
        weight_bounds = [self.model.weight_min, self.model.weight_max]
        data_buffer = {
            "observation_buffer": self.observation_buffer,
            "action_buffer": self.action_buffer,
        }

        cost_function = lambda weights: self.objective(data_buffer, weights=weights)

        if constraint_functions:
            constraints = [
                sp.optimize.NonlinearConstraint(constraint, -np.inf, 0,)
                for constraint in constraint_functions
            ]

        optimized_weights = self.optimizer.optimize(
            cost_function, weights_init, weight_bounds, constraints=constraints,
        )
        return optimized_weights

    def _CasADi_update(self, constraint_functions=[]):

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

        if constraint_functions:
            constraints = self.create_constraints(constraint_functions, symbolic_var)

        optimized_weights = self.optimizer.optimize(
            cost_function,
            weights_init,
            weight_bounds,
            constraints=constraints,
            decision_variable_symbolic=symbolic_var,
        )

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


class CriticValue(Critic):
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

        Jc = 0

        for k in range(self.data_buffer_size - 1, -1, -1):
            observation_old = observation_buffer[k - 1, :]
            observation_next = observation_buffer[k, :]
            action_old = action_buffer[k - 1, :]

            # Temporal difference

            critic_old = self.model(observation_old, weights=weights)
            critic_next = self.model(observation_next, use_stored_weights=True)

            e = (
                critic_old
                - self.discount_factor * critic_next
                - self.running_objective(observation_old, action_old)
            )

            Jc += 1 / 2 * e ** 2

        return Jc


class CriticActionValue(Critic):
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
            observation_old = observation_buffer[k - 1, :]
            observation_next = observation_buffer[k, :]
            action_old = action_buffer[k - 1, :]
            action_next = action_buffer[k, :]

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


class CriticCALF(CriticValue):
    def __init__(
        self, *args, safe_decay_rate=1e-4, predictor=[], observation_init=[], **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.safe_decay_rate = safe_decay_rate
        self.predictor = predictor
        self.observation_last_good = observation_init
        self.lb_constraint_violations = []
        self.ub_constraint_violations = []
        self.stabilizing_constraint_violations = []

    def CALF_decay_constraint(self, weights, observation=None):
        critic_prev = self.model(self.observation_last_good, use_stored_weights=True)
        critic_curr = self.model(observation, weights=weights)
        self.stabilizing_constraint_violation = (
            critic_curr
            - critic_prev
            + self.predictor.pred_step_size * self.safe_decay_rate
        )
        return self.stabilizing_constraint_violation

    def CALF_critic_lower_bound_constraint(self, weights, observation=None):
        self.lb_constraint_violation = 1e-3 * rc.norm_2(observation) - self.model(
            observation, weights=weights
        )
        return self.lb_constraint_violation

    def CALF_critic_upper_bound_constraint(self, weights, observation=None):
        self.ub_constraint_violation = self.model(
            observation, weights=weights
        ) - 1e3 * rc.norm_2(observation)
        return self.ub_constraint_violation

    def update(
        self, *args, constraint_functions=[], observation=None, action=None, **kwargs
    ):
        stabilizing_constraints = [
            partial(self.CALF_decay_constraint, observation=observation),
            partial(self.CALF_critic_lower_bound_constraint, observation=observation),
            partial(self.CALF_critic_upper_bound_constraint, observation=observation),
        ]

        self.weights_acceptance_status = super().update(
            constraint_functions=stabilizing_constraints,
            observation=observation,
            **kwargs
        )

        self.stabilizing_constraint_violations.append(
            self.stabilizing_constraint_violation
        )
        self.lb_constraint_violations.append(self.lb_constraint_violation)
        self.ub_constraint_violations.append(self.ub_constraint_violation)

        if self.weights_acceptance_status == "accepted":
            self.observation_last_good = observation
            return True
        else:
            return False


class CriticTrivial(Critic):
    """
    This is a dummy to calculate outcome (accumulated running objective).

    """

    def __init__(self, running_objective, sampling_time=0.01):
        self.running_objective = running_objective
        self.sampling_time = sampling_time
        self.outcome = 0
        self.model = ModelWeightContainer()
        self.model.weights = None

    def __call__(self):
        return self.outcome

    def objective(self, weights):
        pass

    def get_optimized_weights(self, constraint_functions=[], time=None):
        pass

    def update_buffers(self, observation, action):
        self.update_outcome(observation, action)

    def update(self, constraint_functions=[], time=None):
        pass

    def update_outcome(self, observation, action):

        self.outcome += self.running_objective(observation, action) * self.sampling_time

    def reset(self):
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
        action = self.actor_model.weights[observation]
        if tuple(self.terminal_state) == observation:
            return self.running_objective(observation, action)

        return self.objective(observation, action)

    def update(self):
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
        return (
            self.running_objective(observation, action)
            + self.discount_factor
            * self.model.weights[self.predictor.predict(observation, action)]
        )


class CriticTabularPI(CriticTabularVI):
    def __init__(self, *args, tolerance=1e-3, N_update_iters_max=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance = tolerance
        self.N_update_iters_max = N_update_iters_max

    def update(self):
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
