# TODO: EXTEND DOCSTRING
"""
This module contains optimization routines to be used in optimal controllers, policies, critics etc.

"""
import rcognita.base
from rcognita.__utilities import rc
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import numpy as np
import warnings

try:
    from casadi import vertcat, nlpsol, DM, MX, Function, Opti

except (ModuleNotFoundError, ImportError):
    pass

from abc import ABC, abstractmethod
import time

try:
    import torch.optim as optim
    import torch
    from torch.utils.data import Dataset, DataLoader
    from rcognita.data_buffers import UpdatableSampler

except ModuleNotFoundError:
    from unittest.mock import MagicMock

    torch = MagicMock()
    UpdatableSampler = MagicMock()

from rcognita.callbacks import apply_callbacks
from typing import Callable, Optional, Union, Any
from functools import partial
from inspect import signature
from dataclasses import dataclass, field
from .__utilities import TORCH, CASADI, NUMPY, type_inference
from .base import RcognitaBase


class Optimizer:
    pass


def partial_positionals(func, positionals, **keywords):
    def wrapper(*args, **kwargs):
        arg = iter(args)
        return func(
            *(
                positionals[i] if i in positionals else next(arg)
                for i in range(len(args) + len(positionals))
            ),
            **{**keywords, **kwargs},
        )

    return wrapper


@dataclass
class OptimizerConfig:
    kind: str
    opt_method: Optional[Any] = None
    opt_options: dict = field(default_factory=lambda: {})
    log_options: dict = field(default_factory=lambda: {})
    config_options: dict = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        self.__dict__.update(self.config_options)


@dataclass(frozen=True)
class Constraint:
    kind: str
    func: Callable
    params: dict
    dvar_idx: int = 0

    def __call__(self, dvar, **params):
        _params = list(map(params.get, list(self.params.keys())))
        return self.func(dvar, *_params)


# TODO: DOCSTRING, RENAME INTO OPTIMIZER
class Optimizable(RcognitaBase):
    def __init__(self, optimizer_config: OptimizerConfig) -> None:
        self.optimizer_config = optimizer_config
        self.kind = optimizer_config.kind
        if self.kind == "symbolic":
            self.__opti = Opti()
            self.__variables = []
            self.__parameters = []
            self.__opt_func = None
            self.__initial_guess_form = None
            if optimizer_config.opt_method is None:
                optimizer_config.opt_method = "ipopt"
        elif self.kind == "numeric":
            self.__N_objective_args = None
            self.__decision_var_idx = None
            self.__obj_input_encoding = None
            self.__constraint_input_encodings = None
            if optimizer_config.opt_method is None:
                optimizer_config.opt_method = "SLSQP"
        elif self.kind == "tensor":
            self.__decision_variable = None
            self.optimizer = None
            if optimizer_config.opt_method is None:
                from torch.optim import Adam

                optimizer_config.opt_method = Adam
        else:
            raise NotImplementedError("Not implemented this kind of optimizer")

        self.opt_method = optimizer_config.opt_method
        self.__constraints = []
        self.__bounds = []
        self.__objective_kind = None
        self.__constraints_kind = None
        self.__opt_options = optimizer_config.opt_options
        self.__log_options = optimizer_config.log_options

    def register_decision_variable(self, dvar):
        self.__decision_variable = dvar

    def register_objective(self, func, *args, decision_var_idx=0, **kwargs):
        if self.kind == "symbolic":
            self.__register_symbolic_objective(func, *args, **kwargs)
        elif self.kind == "numeric":
            self.__register_numeric_objective(func, decision_var_idx)
        elif self.kind == "tensor":
            self.__register_tensor_objective(func, decision_var_idx)

    def __register_symbolic_objective(self, func, *args, **kwargs):
        self._objective = func(*args, **kwargs)
        self.__opti.minimize(self._objective)
        self.__objective_kind = "symbolic"

    def __register_numeric_objective(self, func, decision_var_idx):
        assert callable(func), "objective_function must be callable"
        f_args = func.__code__.co_varnames
        assert decision_var_idx < len(
            f_args
        ), "decision variable index must be less than number of decision variables"
        self._objective = func
        self.__decision_var_idx = decision_var_idx
        self.__N_objective_args = len(f_args)
        self.__obj_input_encoding = [f"arg_{i}" for i in range(self.__N_objective_args)]
        self.__obj_input_encoding[self.__decision_var_idx] = "dvar"
        self.__objective_kind = "numeric"

    def __register_tensor_objective(self, func, decision_var_idx):
        f_args = func.__code__.co_varnames
        assert decision_var_idx < len(
            f_args
        ), "decision variable index must be less than number of decision variables"
        self._objective = func
        self.__decision_var_idx = decision_var_idx
        self.__N_objective_args = len(f_args)
        self.__obj_input_encoding = [f"arg_{i}" for i in range(self.__N_objective_args)]
        self.__obj_input_encoding[self.__decision_var_idx] = "dvar"
        self.__objective_kind = "tensor"

    @staticmethod
    def handle_bounds(
        bounds: Union[list, np.ndarray, None],
        dim_variable: int,
        tile_parameter: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if bounds is None:
            assert dim_variable is not None, "Dimension of the action must be specified"
            bounds = np.array([[-np.inf, np.inf] for _ in range(dim_variable)])
        else:
            assert isinstance(
                bounds, (list, np.ndarray)
            ), "action_bounds must be a list or ndarray"
            if not isinstance(bounds, np.ndarray):
                bounds = np.array(bounds)
            assert len(bounds.shape) == 2, (
                f"action_bounds must be of shape ({dim_variable}, 2)."
                + f" You have ({bounds.shape[0]}, {bounds.shape[1]}"
            )
            assert bounds.shape[0] == dim_variable, (
                f"Action bounds should be of size ({dim_variable}, 2)."
                + f" You have ({bounds.shape[0]}, {bounds.shape[1]})."
            )

        for i, row in enumerate(bounds):
            if row[0] > row[1]:
                raise ValueError(
                    "The lower bound of action is greater"
                    + " than the upper bound of action "
                    + f"at index {i} ({row[0] > row[1]})"
                )
        variable_min = bounds[:, 0]
        variable_max = bounds[:, 1]
        variable_initial_guess = (variable_min + variable_max) / 2
        if tile_parameter > 0:
            variable_sequence_initial_guess = rc.rep_mat(
                variable_initial_guess, 1, tile_parameter
            )
            action_sequence_min = rc.rep_mat(variable_min, 1, tile_parameter)
            action_sequence_max = rc.rep_mat(variable_max, 1, tile_parameter)
            result_bounds = np.array([action_sequence_min, action_sequence_max])
            variable_initial_guess = variable_sequence_initial_guess
        return result_bounds, variable_initial_guess, variable_min, variable_max

    def register_bounds(self, bounds, var=None):
        if self.kind == "symbolic":
            self.__register_symbolic_bounds(bounds, var)

        elif self.kind == "numeric":
            self.__register_numeric_bounds(bounds)

        elif self.kind == "tensor":
            self.__register_tensor_bounds(bounds)

    def __register_symbolic_bounds(self, bounds, var=None):
        if var is None:
            assert (
                self.__variables is not None
            ), "At least one decision variable must be specified!"
            var = self.__variables[0]

        self.__bounds = bounds

        def lb_constr(var):
            return bounds[:, 0] - var

        def ub_constr(var):
            return var - bounds[:, 1]

        self.register_constraint(lb_constr, var)
        self.register_constraint(ub_constr, var)

    def __register_numeric_bounds(self, bounds):
        self.__bounds = Bounds(
            bounds[:, 0],
            bounds[:, 1],
            keep_feasible=True,
        )

    def __register_tensor_bounds(self, bounds):
        self.__bounds = bounds

    def register_constraint(self, func, *args, **kwargs):
        if self.kind == "symbolic":
            self.__register_symbolic_constraint(func, *args, **kwargs)
        elif self.kind == "numeric":
            self.__register_numeric_constraint(func)
        elif self.kind == "tensor":
            self.__register_tensor_constraint(func, *args, **kwargs)

    def __register_symbolic_constraint(self, func, *args, **kwargs):
        constr = func(*args, **kwargs) <= 0
        self.__opti.subject_to(constr)
        self.__constraints.append(constr)
        self.__constraints_kind = "symbolic"

    def __register_numeric_constraint(self, func):
        self.__constraints.append(NonlinearConstraint(func, -np.inf, 0))
        self.__constraints_kind = "numeric"

    def __register_tensor_constraint(self, func, *args, **kwargs):
        assert callable(func), "constraint function must be callable"
        self.__constraints.append(func)
        self.__constraints_kind = "tensor"

    def variable(self, dim):
        assert (
            self.kind == "symbolic"
        ), "Variables definition only available for symbolic optimization."
        var = self.__opti.variable(dim)
        self.__variables.append(var)
        return var

    def parameter(self, dim):
        assert (
            self.kind == "symbolic"
        ), "Parameters definition only available for symbolic optimization."
        par = self.__opti.parameter(dim)
        self.__parameters.append(par)
        return par

    def del_variable(self, idx):
        assert (
            self.kind == "symbolic"
        ), "Variables definition only available for symbolic optimization."
        self.__variables.pop(idx)

    def del_parameter(self, idx):
        assert (
            self.kind == "symbolic"
        ), "Parameters definition only available for symbolic optimization."
        self.__parameters.pop(idx)

    @property
    def defined_params(self):
        assert (
            self.kind == "symbolic"
        ), "Parameters definition only available for symbolic optimization."
        return self.__parameters

    @property
    def defined_vars(self):
        assert (
            self.kind == "symbolic"
        ), "Variables definition only available for symbolic optimization."
        return self.__variables

    @property
    def summary_symbolic(self):
        return {
            "kind": self.kind,
            "variables": self.__variables,
            "parameters": self.__parameters,
            "constraints": self.__constraints,
            "bounds": self.__bounds,
            "objective": self._objective,
        }

    def form_initial_guess(self, *args):
        if self.kind == "symbolic":
            self.__initial_guess_form = list(args)
        else:
            raise NotImplementedError

    @property
    def initial_guess_form(self):
        if self.__initial_guess_form is None:
            self.form_initial_guess(*self.__variables, *self.__parameters)
        return self.__initial_guess_form

    def recreate_constraints(self, *constraints):
        for c in constraints:
            self.register_constraint(c)

    def optimize(self, *args, raw=True, **kwargs):
        result = None
        if self.kind == "symbolic":
            result = self._optimize_symbolic(*args, **kwargs)
            if raw:
                result = result["d_vars"][0].full().reshape(-1)
        elif self.kind == "numeric":
            result = self._optimize_numeric(*args, **kwargs)
            if raw:
                result = result[0]
        elif self.kind == "tensor":
            self._optimize_tensor(*args, **kwargs)
            result = self.__decision_variable
        else:
            raise NotImplementedError

        return result

    def _optimize_symbolic(self, *args, **kwargs):
        if self.__initial_guess_form is None:
            self.form_initial_guess(*self.__variables, *self.__parameters)
        if self.__opt_func is None:
            self.__opti.solver(self.opt_method, self.__log_options, self.__opt_options)
            self.__opt_func = self.__opti.to_function(
                "min_fun",
                self.__variables + self.__parameters,
                [*self.__variables, self._objective],
            )
        result = self.__opt_func(*args, **kwargs)
        return {"d_vars": result[:-1], "objective": result[-1]}

    def _optimize_numeric(self, initial_guess, *args):
        if len(args) > 0:
            assert (
                self.__N_objective_args is not None
                and self.__N_objective_args == len(args) + 1
            ), "Wrong number of arguments or objective function not defined."
            arg_idxs = list(range(self.__N_objective_args))
            assert (
                self.__decision_var_idx is not None
            ), "Something went wrong, check your arguments"
            arg_idxs.remove(self.__decision_var_idx)
            objective = partial_positionals(
                self._objective, {arg_idxs[i]: x for i, x in enumerate(args)}
            )
        else:
            objective = self._objective
        opt_result = minimize(
            objective,
            x0=initial_guess,
            method=self.opt_method,
            bounds=self.__bounds,
            options=self.__opt_options,
            constraints=self.__constraints,
            tol=1e-7,
        )
        return opt_result.x, opt_result

    def _optimize_tensor(self, dataset):
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=self.optimizer_config.config_options["shuffle"],
            batch_size=self.optimizer_config.config_options["batch_size"]
            if self.optimizer_config.config_options["batch_size"] is not None
            else len(dataset),
        )
        if self.optimizer is None:
            assert (
                self.__decision_variable is not None
            ), "Optimization parameters not defined."
            assert self.opt_method is not None and callable(
                self.opt_method
            ), f"Wrong optimization method {self.opt_method}."
            self.optimizer = self.opt_method(
                self.__decision_variable, **self.__opt_options
            )
        batch_sample = next(iter(dataloader))
        self.optimizer.zero_grad()
        objective_value = self._objective(batch_sample)
        objective_value.backward()
        self.optimizer.step()


torch_default_config = OptimizerConfig(
    kind="tensor",
    opt_options={"lr": 1e-3},
    config_options={"batch_size": 500, "shuffle": False, "iterations": 30},
)
casadi_default_config = OptimizerConfig(
    kind="symbolic",
    opt_options={"print_level": 0},
    log_options={"print_in": False, "print_out": False, "print_time": True},
    opt_method="ipopt",
)
scipy_default_config = OptimizerConfig(
    kind="numeric",
)


# TODO: WHTA IS THIS? NEEDED?
class TorchDataloaderOptimizer:
    """
    Optimizer class that uses PyTorch as its optimization engine.
    """

    engine = "Torch"

    def __init__(
        self,
        opt_options,
        model,
        shuffle=True,
        opt_method=None,
        batch_size=None,
        sheduler_method=None,
        sheduler_options=None,
        batch_sampler=None,
        verbose=False,
    ):
        """
        Initialize an instance of TorchOptimizer.

        :param opt_options: Options for the PyTorch optimizer.
        :type opt_options: dict
        :param iterations: Number of iterations to optimize the model.
        :type iterations: int
        :param opt_method: PyTorch optimizer class to use. If not provided, Adam is used.
        :type opt_method: torch.optim.Optimizer
        :param verbose: Whether to print optimization progress.
        :type verbose: bool
        """
        if opt_method is None:
            opt_method = torch.optim.Adam

        self.opt_method = opt_method
        self.opt_options = opt_options
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.model = model
        self.optimizer = self.opt_method(self.model.parameters(), **self.opt_options)
        self.sheduler_method = sheduler_method
        self.sheduler_options = sheduler_options
        self.sheduler = self.sheduler_method(self.optimizer, **self.sheduler_options)

        if isinstance(batch_sampler, UpdatableSampler):
            self.batch_sampler = batch_sampler
        else:
            self.batch_sampler = None

    @apply_callbacks()
    def post_epoch(self, idx_epoch, last_epoch_objective):
        return idx_epoch, last_epoch_objective

    def optimize(self, objective, dataset):  # remove model and add parameters instead
        """
        Optimize the model with the given objective.

        :param objective: Objective function to optimize.
        :type objective: callable
        :param model: Model to optimize.
        :type model: torch.nn.Module
        :param model_input: Inputs to the model.
        :type model_input: torch.Tensor
        """

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size if self.batch_size is not None else len(dataset),
        )

        batch_sample = next(iter(dataloader))
        self.optimizer.zero_grad()
        objective_value = objective(batch_sample)
        last_epoch_objective = objective_value.item()
        objective_value.backward()
        self.optimizer.step()
        self.sheduler.step()

        self.post_epoch(1, last_epoch_objective)


# TODO: REMOVE
class TorchProjectiveOptimizer:
    """
    Optimizer class that uses PyTorch as its optimization engine.
    """

    engine = "Torch"

    def __init__(
        self,
        bounds,
        opt_options,
        prediction_horizon=0,
        iterations=1,
        opt_method=None,
        verbose=False,
    ):
        """
        Initialize an instance of TorchOptimizer.

        :param opt_options: Options for the PyTorch optimizer.
        :type opt_options: dict
        :param iterations: Number of iterations to optimize the model.
        :type iterations: int
        :param opt_method: PyTorch optimizer class to use. If not provided, Adam is used.
        :type opt_method: torch.optim.Optimizer
        :param verbose: Whether to print optimization progress.
        :type verbose: bool
        """
        self.bounds = bounds
        if opt_method is None:
            opt_method = torch.optim.Adam
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.iterations = iterations
        self.verbose = verbose
        self.loss_history = []
        self.action_size = self.bounds[:, 1].shape[0]
        self.upper_bound = torch.squeeze(
            torch.tile(torch.tensor(self.bounds[:, 1]), (1, prediction_horizon + 1))
        )

    def optimize(self, *model_input, objective, model):
        """
        Optimize the model with the given objective.

        :param objective: Objective function to optimize.
        :type objective: callable
        :param model: Model to optimize.
        :type model: torch.nn.Module
        :param model_input: Inputs to the model.
        :type model_input: torch.Tensor
        """
        optimizer = self.opt_method([model_input[0]], **self.opt_options)
        # optimizer.zero_grad()

        for _ in range(self.iterations):
            optimizer.zero_grad()
            loss = objective(*model_input)
            # loss_before = loss.detach().numpy()
            loss.backward()
            optimizer.step()
            for param in [model_input[0]]:
                param.requires_grad = False
                param /= self.upper_bound
                param.clamp_(-1, 1)
                param *= self.upper_bound
                param.requires_grad = True
            # optimizer.zero_grad()
            # loss_after = objective(*model_input).detach().numpy()
            # print(loss_before - loss_after)
            if self.verbose:
                print(objective(*model_input))
        # self.loss_history.append([loss_before, loss_after])
        model.weights = torch.nn.Parameter(model_input[0][: self.action_size])
        return model_input[0]


# TODO: REMOVE?
class BruteForceOptimizer:
    """
    Optimizer that searches for the optimal solution by evaluating all possible variants in parallel."
    """

    engine = "bruteforce"

    def __init__(self, possible_variants, N_parallel_processes=0):
        """
        Initialize an instance of BruteForceOptimizer.

        :param N_parallel_processes: number of processes to use in parallel
        :type N_parallel_processes: int
        :param possible_variants: list of possible variants to evaluate
        :type possible_variants: list
        """
        self.N_parallel_processes = N_parallel_processes
        self.possible_variants = possible_variants

    def element_wise_maximization(self, x):
        """
        Find the variant that maximizes the reward for a given element.

        :param x: element to optimize
        :type x: tuple
        :return: variant that maximizes the reward
        :rtype: int
        """
        reward_function = lambda variant: self.objective(variant, x)
        reward_function = np.vectorize(reward_function)
        values = reward_function(self.possible_variants)
        return self.possible_variants[np.argmax(values)]

    def optimize(self, objective, weights):
        """
        Maximize the objective function over the possible variants.

        :param objective: The objective function to maximize.
        :type objective: Callable
        :param weights: The weights to optimize.
        :type weights: np.ndarray
        :return: The optimized weights.
        :rtype: np.ndarray
        """
        self.weights = weights
        self.objective = objective
        indices = tuple(
            [(i, j) for i in range(weights.shape[0]) for j in range(weights.shape[1])]
        )
        for x in indices:
            self.weights[x] = self.element_wise_maximization(x)

        return self.weights

        # with Pool(self.n_pools) as p:
        #     result_weights = p.map(
        #         self.element_wise_maximization,
        #         np.nditer(self.weights, flags=["external_loop"]),
        #     )[0]
        # return result_weights
