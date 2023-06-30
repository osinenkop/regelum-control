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
from typing import Callable, Optional, Union, Any, List, Iterable
from functools import partial
from inspect import signature
from dataclasses import dataclass, field
from .__utilities import TORCH, CASADI, NUMPY, type_inference
from .base import RcognitaBase
from functools import lru_cache
from collections.abc import Mapping


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
    metadata: Any = None
    is_constant: bool = False

    def renamed(self, new_name):
        return Variable(
            name=new_name,
            dims=self.dims,
            data=self.data,
            metadata=self.metadata,
            is_constant=self.is_constant,
        )

    def with_data(self, data):
        return Variable(
            name=self.name,
            dims=self.dims,
            data=data,
            metadata=self.metadata,
            is_constant=self.is_constant,
        )

    def as_constant(self):
        if not self.is_constant:
            return Variable(
                name=self.name,
                dims=self.dims,
                data=self.data,
                metadata=self.metadata,
                is_constant=True,
            )
        else:
            return self

    def to_metadata_dict(self):
        return {self.name: self.metadata}

    def to_data_dict(self):
        return {self.name: self.data}

    def to_dims_dict(self):
        return {self.name: self.dims}

    def __radd__(self, other):
        return self

    def __add__(self, other):
        if isinstance(other, Variable):
            return VarContainer([self, other])
        elif isinstance(other, VarContainer):
            return VarContainer([self] + other.variables)


@dataclass(frozen=True)
class VarContainer(Mapping):
    _variables: Iterable[Variable]

    def __post_init__(self):
        super().__setattr__("_variables", tuple(getattr(self, "_variables")))
        super().__setattr__(
            "_variables_hashmap", {var.name: var for var in getattr(self, "_variables")}
        )

    def to_dict(self):
        return self._variables_hashmap

    def set_data(self, **kwargs):
        return VarContainer(
            [
                var.with_data(kwargs.get(var.name)) if var.name in kwargs else var
                for var in self.variables
            ]
        )

    @property
    def variables(self):
        return self._variables

    @property
    def constants(self):
        return VarContainer(
            tuple(variable for variable in self.variables if variable.is_constant)
        )

    @property
    def decision_variables(self):
        return VarContainer(
            tuple(variable for variable in self.variables if not variable.is_constant)
        )

    def __radd__(self, other):
        return self

    def __add__(self, other):
        if isinstance(other, VarContainer):
            return VarContainer(other.variables + self.variables)
        elif isinstance(other, Variable):
            return VarContainer(self.variables + (other,))

    def __iter__(self):
        for variable in self.variables:
            yield variable

    def __len__(self):
        return len(self.to_dict())

    def __getitem__(self, key):
        if isinstance(key, int):
            assert 0 <= key < len(self.variables), f"Index {key} is out of bounds."
            return self.variables[key]
        elif isinstance(key, slice):
            return VarContainer(self.variables[key])
        elif isinstance(key, str):
            res = self._variables_hashmap.get(key)
            assert res is not None, f"Variable {key} not found."
            return res

    @property
    @lru_cache
    def names(self):
        return tuple(var.name for var in self.variables)

    @property
    @lru_cache
    def metadatas(self):
        return tuple(var.metadata for var in self.variables)

    def to_data_dict(self):
        return {var.name: var.data for var in self.variables}

    @property
    def with_data(self):
        return {name: data for name, data in self.to_data_dict().items()}


@dataclass
class FunctionWithSignature:
    func: Callable
    parameters: VarContainer = field(default_factory=lambda: VarContainer([]))
    is_objective: bool = False

    def __post_init__(self) -> None:
        self.__signature = self.__parse_signature(self.func)
        parameter_names = set(self.parameters.names)
        kwargs_intersection = parameter_names & set(self.__signature)
        if kwargs_intersection != parameter_names:
            raise ValueError(
                "Unknown parameters encountered: "
                f"{self.parameters.keys() - kwargs_intersection}"
            )

        self.name = self.func.__name__
        self.await_parameters = False

    def __call__(self, *args, **kwargs):
        """
        Call the function with the given keyword arguments.
        Only keyword arguments that are set will be passed to the function.

        :param kwargs: The keyword arguments to be passed to the function.
        :type kwargs: dict
        :return: The return value of the function.
        :raises ValueError: If not all required parameters have been set.
        :rtype: Any
        """
        if kwargs == {} and len(args) == 1:
            return self.func(**{self.free_placeholders[0]: args[0]})
        if not self.await_parameters:
            kwargs_to_pass = {
                k: v for k, v in kwargs.items() if k in self.free_placeholders
            }
            return self.func(**{**kwargs_to_pass, **self.parameters.to_data_dict()})
        else:
            raise ValueError("Not all parameters were set")

    @staticmethod
    @lru_cache
    def __cached_signature(signature) -> tuple:
        return tuple(signature)

    @property
    def signature(self) -> tuple:
        return self.__cached_signature(self.__signature)

    @staticmethod
    @lru_cache
    def __cached_occupied(parameters) -> tuple:
        return tuple(parameters.constants.names)

    @property
    def occupied(self) -> tuple:
        return self.__cached_occupied(self.parameters)

    @staticmethod
    @lru_cache
    def __cached_free_placeholders(signature, default_parameters) -> tuple:
        signature_set = set(signature)
        default_keys = {
            name: data
            for name, data in default_parameters.to_data_dict().items()
            if data is not None
        }.keys()
        return tuple(signature_set - default_keys)

    @property
    def free_placeholders(self) -> tuple:
        """
        Returns a list of free placeholders of the current function.
        Free placeholders are the arguments that
        are not defaulted and do not have a corresponding value.
        This method uses the signature of the function
        and the default parameters keys to determine the free placeholders.

        :return: A list of free placeholders of the current function.
        :rtype: list
        """
        return self.__cached_free_placeholders(self.__signature, self.parameters)

    def declare_parameters(self, parameters: VarContainer) -> None:
        """
        Declare new parameter names for the function.

        :param param_names: A list of string containing the names
        of the new parameters to be declared.
        :type param_names: list[str]
        :raises AssertionError: Raised if unknown parameter is passed.
        """
        assert all(
            [name in self.__signature for name in parameters.names]
        ), "Unknown parameters"
        assert all(
            [parameter.is_constant for parameter in parameters]
        ), "Parameters should be constant variables"
        if set(self.parameters.constants.names) != set(parameters.constants.names):
            self.parameters = parameters
            self.await_parameters = True

    def set_parameters(self, **kwargs) -> None:
        """
        Sets the parameters of the function and returns
        a new function object with the updated parameters.

        Args:
            **kwargs: A dictionary of key-value pairs where
            the keys are parameter names and the values
                are their respective values.

        Raises:
            AssertionError: If the keys of the kwargs dictionary
            are not equal to the parameters of the function.
            ValueError: If unknown parameters are encountered.

        Returns:
            resulting function object
        """
        assert kwargs.keys() == set(self.parameters.names), "Wrong parameters passed"

        kwargs_intersection = kwargs.keys() & self.occupied
        if kwargs_intersection != kwargs.keys():
            raise ValueError(
                f"Unknown parameters encountered: {kwargs.keys() - kwargs_intersection}"
            )
        self.parameters = self.parameters.set_data(**kwargs)
        if set(self.parameters.constants.names) == set(
            self.parameters.with_data.keys()
        ):
            self.await_parameters = False

    def __parse_signature(self, func: Callable) -> List[str]:
        signature_list = []

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
            signature_list.append(param.name)

        return tuple(signature_list)

    def __radd__(self, other):
        return self

    def __add__(self, other):
        if isinstance(other, FunctionWithSignature):
            return FuncContainer([self, other])
        elif isinstance(other, FuncContainer):
            return FuncContainer((self,) + other.functions)


@dataclass(frozen=True)
class FuncContainer(Mapping):
    _functions: Iterable[FunctionWithSignature]

    def __post_init__(self):
        super().__setattr__("_functions", tuple(getattr(self, "_functions")))
        super().__setattr__(
            "_functions_hashmap", {var.name: var for var in getattr(self, "_functions")}
        )

    def to_dict(self):
        return self._functions_hashmap

    @property
    def functions(self):
        return self._functions

    @property
    def objectives(self):
        return FuncContainer(
            tuple(function for function in self.functions if function.is_objective)
        )

    @property
    def constraints(self):
        return FuncContainer(
            tuple(function for function in self.functions if not function.is_objective)
        )

    def __radd__(self, other):
        return self

    def __add__(self, other):
        if isinstance(other, FuncContainer):
            return FuncContainer(other.functions + self.functions)
        elif isinstance(other, FunctionWithSignature):
            return FuncContainer(self.functions + (other,))

    def __iter__(self):
        for function in self.functions:
            yield function

    def __len__(self):
        return len(self.to_dict())

    def __getitem__(self, key):
        if isinstance(key, int):
            assert 0 <= key < len(self.functions), f"Index {key} is out of bounds."
            return self.functions[key]
        elif isinstance(key, slice):
            return VarContainer(self.functions[key])
        elif isinstance(key, str):
            res = self._functions_hashmap.get(key)
            assert res is not None, f"Function {key} not found."
            return res

    @property
    def names(self):
        return tuple(function.name for function in self.functions)


# TODO: DOCSTRING
class Optimizable(RcognitaBase):
    def __init__(self, optimizer_config: OptimizerConfig) -> None:
        self.optimizer_config = optimizer_config
        self.kind = optimizer_config.kind
        self.__is_problem_defined = False
        self.__parameters = VarContainer([])
        self.__variables = VarContainer([])
        self.__functions = FuncContainer([])

        if self.kind == "symbolic":
            self.__opti = Opti()
            self.__opt_func = None
            if optimizer_config.opt_method is None:
                optimizer_config.opt_method = "ipopt"
        elif self.kind == "numeric":
            if optimizer_config.opt_method is None:
                optimizer_config.opt_method = "SLSQP"
        elif self.kind == "tensor":
            self.optimizer = None
            if optimizer_config.opt_method is None:
                from torch.optim import Adam

                optimizer_config.opt_method = Adam
        else:
            raise NotImplementedError("Not implemented this kind of optimizer")

        self.opt_method = optimizer_config.opt_method
        self.__opt_options = optimizer_config.opt_options
        self.__log_options = optimizer_config.log_options

    @property
    def objectives(self):
        return self.__functions.objectives

    @property
    def constraints(self):
        return self.__functions.constraints

    @property
    def functions(self):
        return self.__functions

    def create_variable(self, *dims, name: str, is_constant=False):
        metadata = None
        if self.kind == "symbolic":
            if len(dims) == 1:
                assert isinstance(
                    dims[0], (int, tuple)
                ), "Dimension must be integer or tuple"
                if isinstance(dims[0], tuple):
                    assert len(dims[0]) <= 2, "Symbolic variable dimension must be <= 2"
                    metadata = (
                        self.__opti.variable(*dims[0])
                        if not is_constant
                        else self.__opti.parameter(*dims[0])
                    )
                else:
                    metadata = (
                        self.__opti.variable(dims[0])
                        if not is_constant
                        else self.__opti.parameter(dims[0])
                    )
            else:
                metadata = (
                    self.__opti.variable(*dims)
                    if not is_constant
                    else self.__opti.parameter(*dims)
                )
        new_variable = Variable(
            name=name, dims=dims, metadata=metadata, is_constant=is_constant
        )
        self.__variables = self.__variables + new_variable
        return new_variable

    def register_objective(self, func: Callable, variables: List[Variable]):
        func = FunctionWithSignature(func, is_objective=True)
        variables = VarContainer(variables)

        if self.kind == "symbolic":
            self.__register_symbolic_objective(
                func,
                variables=variables,
            )
        elif self.kind == "numeric":
            self.__register_numeric_objective(
                func,
                variables=variables.constants,
            )
        elif self.kind == "tensor":
            self.__register_tensor_objective(
                func,
                variables=variables.constants,
            )

        self.__functions = self.__functions + func

    @property
    def variables(self):
        return self.__variables

    def __infer_symbolic_prototype(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        func.declare_parameters(variables.constants)
        func.set_parameters(
            **{k: v.metadata for k, v in variables.constants.to_dict().items()}
        )
        return func(
            **{
                k: v.metadata for k, v in variables.decision_variables.to_dict().items()
            },
        )

    def __register_symbolic_objective(
        self,
        func: FunctionWithSignature,
        variables: VarContainer,
    ):
        self._objective = self.__infer_symbolic_prototype(func, variables)
        self.__opti.minimize(self._objective)

    def __register_numeric_objective(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        assert callable(func), "objective_function must be callable"
        func.declare_parameters(variables.constants)
        self._objective = func

    def __register_tensor_objective(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        assert callable(func), "objective_function must be callable"
        func.declare_parameters(variables.constants)
        self._objective = func

    @staticmethod
    def handle_bounds(
        bounds: Union[list, np.ndarray, None],
        dim_variable: int,
        tile_parameter: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Given bounds for each dimension of a variable, this function returns a tuple of
        the following arrays: the bounds of each action,
        the initial guess for a variable,
        the minimum value of each variable, and the maximum value of each variable.

        :param bounds: A list, numpy array, or None that represents the bounds for each
        dimension of a variable. If None is given,
        bounds will be assumed to be (-inf, inf)
        for each dimension. Otherwise, bounds should have shape (dim_variable, 2),
        where
        dim_variable is the number of dimensions of the variable.
        :type bounds: Union[list, np.ndarray, None]

        :param dim_variable: An integer representing the number of dimensions
        of the variable.
        :type dim_variable: int

        :param tile_parameter: An optional integer that represents
        the number of copies of
        the variable to be made. If tile_parameter is greater than zero,
        variable_initial_guess
        and result_bounds will be tiled with this number of copies.
        :type tile_parameter: int, optional

        :return: A tuple of numpy arrays containing the bounds of each action,
        the initial guess for a variable, the minimum value of each variable,
        and the maximum value of each variable.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
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

    def register_bounds(self, variable_to_bound: Variable, bounds: np.ndarray):
        assert isinstance(
            variable_to_bound, Variable
        ), "variable_to_bound should be of type Variable, "
        f"not {type(variable_to_bound)}"

        if self.kind == "symbolic":
            self.__register_symbolic_bounds(variable_to_bound, bounds)

        elif self.kind == "numeric":
            self.__register_numeric_bounds(bounds)

        elif self.kind == "tensor":
            self.__register_tensor_bounds(bounds)

    def __register_symbolic_bounds(
        self, variable_to_bound: Variable, bounds: np.ndarray
    ):
        self.__bounds = bounds

        def lb_constr(var):
            return bounds[:, 0] - var

        def ub_constr(var):
            return var - bounds[:, 1]

        self.register_constraint(
            lb_constr, variables=[variable_to_bound.renamed("var")]
        )
        self.register_constraint(
            ub_constr, variables=[variable_to_bound.renamed("var")]
        )

    def __register_numeric_bounds(self, bounds):
        self.__bounds = Bounds(
            bounds[:, 0],
            bounds[:, 1],
            keep_feasible=True,
        )

    def __register_tensor_bounds(self, bounds):
        self.__bounds = bounds

    def register_constraint(self, func: Callable, variables: List[Variable]):
        func = FunctionWithSignature(func)
        variables = VarContainer(variables)

        if self.kind == "symbolic":
            self.__register_symbolic_constraint(func, variables)
        elif self.kind == "numeric":
            self.__register_numeric_constraint(func, variables.constants)
        elif self.kind == "tensor":
            self.__register_tensor_constraint(func, variables.constants)

        self.__functions = self.__functions + func

    def __register_symbolic_constraint(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        func = self.__infer_symbolic_prototype(func, variables)
        constr = func <= 0
        self.__opti.subject_to(constr)

    def __register_numeric_constraint(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        func.declare_parameters(variables)

    def __register_tensor_constraint(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        func.declare_parameters(variables.names)

    @property
    def constants(self):
        return self.__variables.constants

    @property
    def decision_variables(self):
        return self.__variables.decision_variables

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

    def substitute_parameters(self, **parameters):
        for function in self.functions:
            function.set_parameters(
                **{
                    k: v
                    for k, v in parameters.items()
                    if k in function.parameters.constants.names
                }
            )

    def optimize(self, raw=False, **parameters):
        if not self.__is_problem_defined:
            self.define_problem()

        result = None
        if self.kind == "symbolic":
            result = self.optimize_symbolic(**parameters, raw=raw)
        elif self.kind == "numeric":
            result = self.optimize_numeric(**parameters, raw=raw)
        elif self.kind == "tensor":
            self.optimize_tensor(**parameters)
            result = self.__decision_variable
        else:
            raise NotImplementedError

        return result

    @property
    def opt_func(self):
        return self.__opt_func

    def optimize_symbolic(self, raw=False, **kwargs):
        # ToDo: add multiple objectives
        if self.__opt_func is None:
            self.__opti.solver(self.opt_method, self.__log_options, self.__opt_options)
            self.__opt_func = self.__opti.to_function(
                "min_fun",
                list(self.variables.metadatas),
                [*self.decision_variables.metadatas, self._objective],
                list(self.variables.names),
                [*self.decision_variables.names, *self.objectives.names],
                {"allow_duplicate_io_names": True},
            )

        result = self.__opt_func(**kwargs)
        return (
            result
            if raw
            else {
                name: value
                for name, value in result.items()
                if name in self.decision_variables.names
            }
        )

    def optimize_numeric(self, raw=False, **parameters):
        self.substitute_parameters(**parameters)
        constraints = [
            NonlinearConstraint(func, -np.inf, 0) for func in self.constraints
        ]
        initial_guess = parameters.get(self.decision_variables.names[0])
        opt_result = minimize(
            self._objective,
            x0=initial_guess,
            method=self.opt_method,
            bounds=self.__bounds,
            options=self.__opt_options,
            constraints=constraints,
            tol=1e-7,
        )
        return opt_result if raw else opt_result.x

    def optimize_tensor(self, **parameters):
        dataloader = parameters.get("dataloader")
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

    def define_problem(self):
        raise NotImplementedError
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
