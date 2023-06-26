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
import inspect


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
class Variable:
    name: str
    dims: tuple
    data: Any = None


@dataclass(frozen=True)
class Parameter:
    name: str
    dims: tuple
    data: Any = None


@dataclass(frozen=True)
class FunctionWithParameters:
    func: Callable
    signature: list
    default_parameters: dict = field(default_factory=lambda: {})

    def __call__(self, **kwargs):
        kwargs.update(self.default_parameters)
        return self.func(**kwargs)


class WithSignature:
    def __init__(self, func: Callable):
        self.__signature = self.__parse_signature(func)
        self.__func = FunctionWithParameters(func, self.__signature)
        self.func = FunctionWithParameters(func, self.__signature)
        self.got_parameters = True
        self.parameters = []

    def declare_parameters(self, param_names):
        assert all(
            [name in self.__signature for name in param_names]
        ), "Unknown parameters"
        if set(self.parameters) != set(param_names):
            self.parameters = param_names
            self.got_parameters = False

    @property
    def signature(self):
        return self.__func.signature

    @property
    def free_placeholders(self):
        return set(self.__signature) - set(self.__func.default_parameters.keys())

    def __parse_signature(self, func: Callable):
        __signature = []

        parameters = (
            func.signature
            if hasattr(func, "signature")
            else signature(func).parameters.values()
        )
        for param in parameters:
            if param.kind == param.VAR_POSITIONAL:
                raise ValueError("Undefined number of arguments")
            if param.kind == param.VAR_KEYWORD:
                raise ValueError("Undefined number of keyword arguments")
            __signature.append(param.name)

        return __signature

    def set_parameters(self, **kwargs):
        assert list(kwargs.keys()) == self.parameters, "Wrong parameters passed"

        kwargs_intersection = kwargs.keys() & set(self.func.signature)
        if kwargs_intersection != kwargs.keys():
            raise ValueError(
                f"Unknown parameters encountered: {kwargs.keys() - kwargs_intersection}"
            )

        new_func = FunctionWithParameters(
            self.func,
            default_parameters=kwargs,
            signature=list(filter(lambda x: x not in kwargs.keys(), self.__signature)),
        )

        self.__func = new_func
        self.got_parameters = True

    def __call__(self, **kwargs):
        if self.got_parameters:
            return self.__func(
                **{k: v for k, v in kwargs.items() if k in self.free_placeholders}
            )
        else:
            raise ValueError("Not all parameters were passed")


# TODO: DOCSTRING
class Optimizable(RcognitaBase):
    def __init__(self, optimizer_config: OptimizerConfig) -> None:
        self.optimizer_config = optimizer_config
        self.kind = optimizer_config.kind
        self.__is_problem_defined = False
        self.__variables = {}
        self.__parameters = {}

        if self.kind == "symbolic":
            self.__opti = Opti()
            self.__variables_symbolic = None
            self.__parameters_symbolic = None
            self.__opt_func = None
            if optimizer_config.opt_method is None:
                optimizer_config.opt_method = "ipopt"
        elif self.kind == "numeric":
            self.__N_objective_args = None
            self.__decision_var_idx = None
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
        self.__opt_options = optimizer_config.opt_options
        self.__log_options = optimizer_config.log_options

    def register_decision_variable(self, dvar):
        self.__decision_variable = dvar

    def register_objective(self, func, **kwargs):
        func = WithSignature(func)
        if self.kind == "symbolic":
            self.__register_symbolic_objective(func, **kwargs)
        elif self.kind == "numeric":
            self.__register_numeric_objective(func, **kwargs)
        elif self.kind == "tensor":
            self.__register_tensor_objective(func, **kwargs)

    def __register_symbolic_objective(self, func, **kwargs):
        self._objective = func(**{k: v.data for k, v in kwargs.items()})
        self.__opti.minimize(self._objective)

    def __register_numeric_objective(self, func, **kwargs):
        assert callable(func), "objective_function must be callable"
        self._objective = func

    def __register_tensor_objective(self, func, **kwargs):
        self._objective = func

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
            ), "bounds must be a list or ndarray"
            if not isinstance(bounds, np.ndarray):
                bounds = np.array(bounds)
            assert len(bounds.shape) == 2, (
                f"bounds must be of shape ({dim_variable}, 2)."
                + f" You have ({bounds.shape[0]}, {bounds.shape[1]}"
            )
            assert bounds.shape[0] == dim_variable, (
                f"bounds should be of size ({dim_variable}, 2)."
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

    def register_bounds(self, bounds):
        if self.kind == "symbolic":
            self.__register_symbolic_bounds(bounds)

        elif self.kind == "numeric":
            self.__register_numeric_bounds(bounds)

        elif self.kind == "tensor":
            self.__register_tensor_bounds(bounds)

    def __register_symbolic_bounds(self, bounds):
        assert (
            self.__variables_symbolic is not None
        ), "Decision variable must be specified!"
        self.__bounds = bounds

        def lb_constr(var):
            return bounds[:, 0] - var

        def ub_constr(var):
            return var - bounds[:, 1]

        self.register_constraint(
            lb_constr, var=self.__variables_symbolic[self.__decision_variable]
        )
        self.register_constraint(
            ub_constr, var=self.__variables_symbolic[self.__decision_variable]
        )

    def __register_numeric_bounds(self, bounds):
        self.__bounds = Bounds(
            bounds[:, 0],
            bounds[:, 1],
            keep_feasible=True,
        )

    def __register_tensor_bounds(self, bounds):
        self.__bounds = bounds

    def register_constraint(self, func, **kwargs):
        func = WithSignature(func)
        if self.kind == "symbolic":
            self.__register_symbolic_constraint(func, **kwargs)
        elif self.kind == "numeric":
            self.__register_numeric_constraint(func)
        elif self.kind == "tensor":
            self.__register_tensor_constraint(func)

    def __register_symbolic_constraint(self, func, **kwargs):
        constr = func(**kwargs) <= 0
        self.__opti.subject_to(constr)
        self.__constraints.append(constr)

    def __register_numeric_constraint(self, func):
        self.__constraints.append(NonlinearConstraint(func, -np.inf, 0))

    def __register_tensor_constraint(self, func):
        assert callable(func), "constraint function must be callable"
        self.__constraints.append(func)

    def variable(self, *dims, name: str):
        self.decision_variable = name
        if self.kind == "symbolic":
            if len(dims) == 1:
                assert isinstance(
                    dims[0], (int, tuple)
                ), "Dimension must be integer or tuple"
                if isinstance(dims[0], tuple):
                    assert len(dims[0]) <= 2, "Symbolic variable dimension must be <= 2"
                    var = self.__opti.variable(*dims[0])
                else:
                    var = self.__opti.variable(dims[0])
            else:
                var = self.__opti.variable(*dims)
            self.__variables[name] = var
            self.__variables_symbolic[name] = var
        else:
            self.__variables[name] = dims

        var = Variable(name, dims, var)
        return var

    def parameter(self, dims, name: str):
        if self.kind == "symbolic":
            if len(dims) == 1:
                assert isinstance(
                    dims[0], (int, tuple)
                ), "Dimension must be integer or tuple"
                if isinstance(dims[0], tuple):
                    assert len(dims[0]) <= 2, "Symbolic variable dimension must be <= 2"
                    var = self.__opti.parameter(*dims[0])
                else:
                    var = self.__opti.parameter(dims[0])
            else:
                var = self.__opti.parameter(*dims)
            self.__parameters[name] = var
            self.__parameters_symbolic[name] = var
            return var
        else:
            self.__parameters[name] = dims
        var = Parameter(name, dims, var)
        return self.__parameters[name]

    @property
    def defined_params(self):
        return self.__parameters

    @property
    def defined_vars(self):
        return self.__variables

    @property
    def summary(self):
        return {
            "kind": self.kind,
            "variables": self.__variables,
            "parameters": self.__parameters,
            "constraints": self.__constraints,
            "bounds": self.__bounds,
            "objective": self._objective,
        }

    def recreate_constraints(self, *constraints):
        for c in constraints:
            self.register_constraint(c)

    def optimize(self, raw=True, **kwargs):
        if not self.__is_problem_defined:
            self.__define_problem()
        self.__pre_optimize(**kwargs)
        result = None
        if self.kind == "symbolic":
            result = self.optimize_symbolic(**kwargs)
            if raw:
                result = result["d_vars"][0].full().reshape(-1)
        elif self.kind == "numeric":
            result = self.optimize_numeric(**kwargs)
            if raw:
                result = result[0]
        elif self.kind == "tensor":
            self.optimize_tensor(**kwargs)
            result = self.__decision_variable
        else:
            raise NotImplementedError

        return result

    def optimize_symbolic(self, *args, **kwargs):
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
        return {"d_vars": result["o0"], "objective": result["o1"]}

    def optimize_numeric(self, initial_guess, *args):
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

    def optimize_tensor(self, dataloader):
        options = self.optimizer_config.config_options
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
        n_epochs = options.get("n_epochs") if options.get("n_epochs") is not None else 1
        for _ in range(n_epochs):
            for batch_sample in dataloader:
                self.optimizer.zero_grad()
                objective_value = self._objective(batch_sample)
                objective_value.backward()
                self.optimizer.step()

    def pre_optimize_symbolic(self):
        pass

    def pre_optimize_numeric(self):
        pass

    def pre_optimize_tensor(self):
        pass

    def __pre_optimize(self):
        if self.kind == "symbolic":
            self.pre_optimize_symbolic()
        elif self.kind == "numeric":
            self.pre_optimize_numeric()
        elif self.kind == "tensor":
            self.pre_optimize_tensor()
        else:
            raise NotImplementedError

    def define_problem_symbolic(self):
        pass

    def define_problem_numeric(self):
        pass

    def define_problem_tensor(self):
        pass

    def __define_problem(self):
        if self.kind == "symbolic":
            self.define_problem_symbolic()
        elif self.kind == "numeric":
            self.define_problem_numeric()
        elif self.kind == "tensor":
            self.define_problem_tensor()

        self.__is_problem_defined = True


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
