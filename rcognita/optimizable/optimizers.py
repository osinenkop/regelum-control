"""Contains main `Optimizable` class.
 
`Optimizable` class is to be used normally as a parent class for all objects that need to be optimized.
"""
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from rcognita.__utilities import rc

try:
    from casadi import Opti

except (ModuleNotFoundError, ImportError):
    pass


try:
    import torch
    from torch.utils.data import DataLoader

    # from rcognita.data_buffers import UpdatableSampler

except ModuleNotFoundError:
    from unittest.mock import MagicMock

    torch = MagicMock()
    UpdatableSampler = MagicMock()

from typing import Callable, List, Tuple, Optional, Union, Dict

from .core.configs import OptimizerConfig
from .core.entities import (
    FunctionWithSignature,
    Hook,
    OptimizationVariable,
    VarContainer,
    FuncContainer,
)
from .core.hooks import requires_grad, detach, data_closure, metadata_closure

import rcognita


class Optimizable(rcognita.RcognitaBase):
    """Base class for all optimizable objects."""

    def __init__(self, optimizer_config: OptimizerConfig) -> None:
        """Initialize an optimizable object."""
        self.optimizer_config = optimizer_config
        self.kind = optimizer_config.kind
        self.__callback_target_events = optimizer_config.callback_target_events
        self.__is_problem_defined = False
        self.__variables: VarContainer = VarContainer([])
        self.__functions: FuncContainer = FuncContainer(tuple())
        self.params_changed = False

        if self.kind == "symbolic":
            self.__opti_common = Opti()
            self.__opti = self.__opti_common.copy()
            self.__opt_func = None
            if optimizer_config.opt_method is None:
                optimizer_config.opt_method = "ipopt"
        elif self.kind == "numeric":
            if optimizer_config.opt_method is None:
                self.__bounds = None
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
    def opt_options(self):
        return self.__opt_options

    @property
    def log_options(self):
        return self.__log_options

    def __recreate_opti(self):
        self.__opti = Opti()

    def __recreate_symbolic_variables(self):
        self.__recreate_opti()

        for variable in self.variables:
            dims = variable.dims
            if isinstance(dims, tuple):
                if len(dims) == 1:
                    dims = dims[0]

            is_constant = variable.is_constant
            if isinstance(dims, tuple):
                metadata = (
                    self.__opti.variable(*dims)
                    if not is_constant
                    else self.__opti.parameter(*dims)
                )
            elif isinstance(dims, int):
                metadata = (
                    self.__opti.variable(dims)
                    if not is_constant
                    else self.__opti.parameter(dims)
                )
            else:
                raise ValueError("Unknown dimensions format")

            variable.with_metadata(metadata, inplace=True)

        self.__recreate_symbolic_functions()

    def __recreate_symbolic_functions(self):
        __functions = sum(
            [
                function
                for function in self.functions
                if not (
                    ("__bound" in function.name) and (function.variables[0].is_constant)
                )
            ]
        )
        for function in __functions:
            function: FunctionWithSignature
            for variable in function.variables:
                if variable.name == "var":
                    variable.metadata = variable.data.metadata

            function = self.__infer_and_register_symbolic_prototype(
                function, function.variables
            )
            metafunc = function.metadata
            if function.is_objective:
                self.__opti.minimize(metafunc)
            else:
                self.__opti.subject_to(metafunc <= 0)

    @property
    def opti(self):
        return self.__opti

    @property
    def objective(self):
        assert len(self.objectives) == 1, "Ambiguous objective definition."
        return self.objectives[0]

    @property
    def objectives(self):
        return self.__functions.objectives

    @property
    def constraints(self):
        return self.__functions.constraints

    @property
    def functions(self):
        return self.__functions

    @property
    def variables(self) -> VarContainer:
        return self.__variables

    def __refresh_binded_variables(self):
        for function in self.functions:
            for variable in function.variables:
                if isinstance(variable.data, OptimizationVariable):
                    variable.is_constant = variable.data.is_constant

    def __fix_variables_tensor(self, variables_to_fix, data_dict, metadata_dict):
        self.__variables.fix(variables_to_fix, hook=Hook(detach, act_on="data"))

    def __fix_variables_symbolic(self, variables_to_fix, data_dict, metadata_dict):
        if metadata_dict is None:
            metadata_dict = {}
        passed_unfixed_variables = sum(self.variables.selected(variables_to_fix))
        assert isinstance(
            passed_unfixed_variables, VarContainer
        ), "An error occured while fixing variables."
        self.__variables.fix(variables_to_fix)
        self.__refresh_binded_variables()
        self.__recreate_symbolic_variables()

        self.params_changed = True

    def fix_variables(
        self,
        variables_to_fix: List[str],
        data_dict: Optional[Dict] = None,
        metadata_dict: Optional[Dict] = None,
    ):
        if self.kind == "tensor":
            self.__fix_variables_tensor(
                variables_to_fix, data_dict=data_dict, metadata_dict=metadata_dict
            )

        elif self.kind == "symbolic":
            self.__fix_variables_symbolic(
                variables_to_fix, data_dict=data_dict, metadata_dict=metadata_dict
            )
        else:
            self.__variables.fix(variables_to_fix)

    def __unfix_variables_tensor(self, variables_to_unfix):
        self.__variables.unfix(
            variables_to_unfix, hook=Hook(requires_grad, act_on="data")
        )

    def __unfix_variables_symbolic(self, variables_to_unfix):
        passed_fixed_variables = sum(self.variables.selected(variables_to_unfix))
        assert isinstance(
            passed_fixed_variables, VarContainer
        ), "An error occured while fixing variables."

        self.__variables.unfix(variables_to_unfix)
        self.__refresh_binded_variables()
        self.__recreate_symbolic_variables()

        self.params_changed = True

    def unfix_variables(
        self,
        variables_to_unfix: List[str],
    ):
        if self.kind == "tensor":
            self.__unfix_variables_tensor(variables_to_unfix=variables_to_unfix)
        elif self.kind == "symbolic":
            self.__unfix_variables_symbolic(variables_to_unfix=variables_to_unfix)
        else:
            self.__variables.unfix(variables_to_unfix)

    def create_variable_metadata(self, *dims, is_constant=False, like=None):
        metadata = None
        if self.kind == "symbolic":
            if like is not None:
                assert hasattr(
                    like, "shape"
                ), "Symbolic variable prototype must have shape"
                dims = like.shape
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
        elif self.kind == "tensor":
            metadata = like
        return metadata

    def create_variable(self, *dims, name: str, is_constant=False, like=None):
        metadata = self.create_variable_metadata(
            *dims, is_constant=is_constant, like=like
        )
        new_variable = OptimizationVariable(
            name=name, dims=dims, metadata=metadata, is_constant=is_constant
        )
        if self.kind == "tensor":
            if like is not None:
                new_variable = new_variable.with_data(like).with_metadata(like)
                new_variable.register_hook(data_closure(like), act_on="data")
                new_variable.register_hook(metadata_closure(like), act_on="metadata")

        self.__variables = self.__variables + new_variable
        return new_variable

    def __infer_and_register_symbolic_prototype(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        metadata = func(
            **variables.to_metadata_dict(), with_metadata=True, raw_eval=True
        )
        func.metadata = metadata
        return func

    def register_objective(
        self,
        func: Callable,
        variables: Union[
            List[OptimizationVariable], VarContainer, OptimizationVariable
        ],
    ):
        func = FunctionWithSignature(func, is_objective=True)
        if isinstance(variables, OptimizationVariable):
            variables = [variables]
        if isinstance(variables, List):
            variables = VarContainer(variables)
        func.declare_variables(variables)

        if self.kind == "symbolic":
            self.__register_symbolic_objective(
                func,
                variables=variables,
            )
        else:
            pass

        new_container = self.__functions + func
        assert new_container is not None, f"Couldn't register objective {func}"
        self.__functions = new_container

    @staticmethod
    def connect_source(
        connect_to: OptimizationVariable,
        func: Callable,
        source: OptimizationVariable,
        act_on="data",
    ):
        def source_hook(whatever):
            return func(source())

        hook = Hook(source_hook, act_on=act_on)
        connect_to.register_hook(hook, first=True)

    def __register_symbolic_objective(
        self,
        func: FunctionWithSignature,
        variables: VarContainer,
    ):
        func = self.__infer_and_register_symbolic_prototype(func, variables)
        self.__opti.minimize(func.metadata)

    @staticmethod
    def handle_bounds(
        bounds: Union[list, np.ndarray, None],
        dim_variable: int,
        tile_parameter: int = 0,
    ) -> Tuple:
        """Given bounds for each dimension of a variable, this function returns a tuple of the following arrays: the bounds of each action,the initial guess for a variable, the minimum value of each variable, and the maximum value of each variable.

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
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        if bounds is None:
            assert dim_variable is not None, "Dimension of the bounds must be specified"
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
            sequence_min = rc.rep_mat(variable_min, 1, tile_parameter)
            sequence_max = rc.rep_mat(variable_max, 1, tile_parameter)
            result_bounds = np.array([sequence_min, sequence_max])
            variable_initial_guess = variable_sequence_initial_guess
        else:
            result_bounds = np.array([variable_min, variable_max])
        return result_bounds, variable_initial_guess, variable_min, variable_max

    def register_bounds(
        self, variable_to_bound: OptimizationVariable, bounds: np.ndarray
    ):
        assert isinstance(
            variable_to_bound, OptimizationVariable
        ), "variable_to_bound should be of type OptimizationVariable, "
        f"not {type(variable_to_bound)}"

        if self.kind == "symbolic":
            self.__register_symbolic_bounds(variable_to_bound, bounds)

        elif self.kind == "numeric":
            self.__register_numeric_bounds(bounds)

        elif self.kind == "tensor":
            self.__register_tensor_bounds(bounds)

    def __register_symbolic_bounds(
        self, variable_to_bound: OptimizationVariable, bounds: np.ndarray
    ):
        self.__bounds = bounds

        def lb_constr(var):
            return bounds[:, 0] - var

        def ub_constr(var):
            return var - bounds[:, 1]

        var = variable_to_bound.renamed("var", inplace=False)
        var.data = variable_to_bound
        self.register_constraint(
            lb_constr, variables=[var], name=f"{variable_to_bound.name}__bound_lower"
        )
        self.register_constraint(
            ub_constr, variables=[var], name=f"{variable_to_bound.name}__bound_upper"
        )

    def __register_numeric_bounds(self, bounds):
        self.__bounds = Bounds(
            bounds[:, 0],
            bounds[:, 1],
            keep_feasible=True,
        )

    def __register_tensor_bounds(self, bounds):
        self.__bounds = bounds

    def register_constraint(
        self, func: Callable, variables: List[OptimizationVariable], name=None
    ):
        func = FunctionWithSignature(func)
        if name is not None:
            func.name = name

        _variables = VarContainer(variables)
        func.declare_variables(_variables)
        if self.kind == "symbolic":
            self.__register_symbolic_constraint(func, _variables)
        elif self.kind == "numeric":
            self.__register_numeric_constraint(func, _variables)
        elif self.kind == "tensor":
            self.__register_tensor_constraint(func, _variables)

        new_container = self.__functions + func
        assert new_container is not None, f"Couldn't register objective {func}"

        self.__functions = new_container

    def __register_symbolic_constraint(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        func = self.__infer_and_register_symbolic_prototype(func, variables)
        constr = func.metadata <= 0
        self.__opti.subject_to(constr)

    def __register_numeric_constraint(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        func.declare_variables(variables)

    def __register_tensor_constraint(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        func.declare_variables(variables)

    @property
    def constants(self):
        return self.__variables.constants

    @property
    def decision_variables(self):
        return self.__variables.decision_variables

    def substitute_parameters(self, **parameters):
        for function in self.functions:
            function.set_parameters(
                **{
                    k: v
                    for k, v in parameters.items()
                    if k in function.variables.constants.names
                }
            )

    def is_target_event(self, event):
        if self.__callback_target_events is None:
            return False

        return event in self.__callback_target_events

    def optimize_on_event(
        self, event: str, *optimize_callback_args, **optimize_callback_kwargs
    ):
        if self.is_target_event(event):
            self.optimize_callback(*optimize_callback_args, **optimize_callback_kwargs)

    def optimize_callback(self, *optimize_callback_args, **optimize_callback_kwargs):
        raise NotImplementedError("optimize_callback is not implemented")

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
            result = self.variables.decision_variables
        else:
            raise NotImplementedError

        return result

    @property
    def opt_func(self):
        return self.__opt_func

    def optimize_symbolic(self, raw=False, **kwargs):
        if self.__opt_func is None or self.params_changed:
            self.__opti.solver(self.opt_method, self.__log_options, self.__opt_options)
            self.__opt_func = self.__opti.to_function(
                "min_fun",
                [variable(with_metadata=True) for variable in self.variables],
                [
                    *[
                        variable(with_metadata=True)
                        for variable in self.decision_variables
                    ],
                    self.objective.metadata,
                ],
                list(self.variables.names),
                [*self.decision_variables.names, *self.objectives.names],
                {"allow_duplicate_io_names": True},
            )
            self.params_changed = False

        for k, v in kwargs.items():
            if k in self.variables:
                if self.variables[k].is_constant:
                    self.__opti.set_value(self.variables[k](with_metadata=True), v)
                else:
                    self.__opti.set_initial(self.variables[k](with_metadata=True), v)

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
            self.objective,
            x0=initial_guess,
            method=self.opt_method,
            bounds=self.__bounds,
            options=self.__opt_options,
            constraints=constraints,
            tol=1e-7,
        )
        return opt_result if raw else opt_result.x

    @apply_callbacks()
    def post_epoch(self, epoch_idx, last_epoch_objective_value):
        return epoch_idx, last_epoch_objective_value

    def optimize_tensor(self, **parameters):
        dataloader = parameters.get("dataloader")
        assert dataloader is not None, "Couldn't find dataloader"
        options = self.optimizer_config.config_options
        if self.optimizer is None:
            assert self.opt_method is not None and callable(
                self.opt_method
            ), f"Wrong optimization method {self.opt_method}."
            dvar = self.variables.decision_variables[0]
            assert dvar is not None, "Couldn't find decision variable"
            assert isinstance(dvar, OptimizationVariable), "Something went wrong..."
            self.optimizer = self.opt_method(dvar(), **self.__opt_options)
        n_epochs = options.get("n_epochs") if options.get("n_epochs") is not None else 1
        assert isinstance(n_epochs, int), "n_epochs must be an integer"
        assert len(self.functions.objectives) == 1, "Only one objective is supported"
        objective = self.functions.objectives[0]
        assert isinstance(objective, FunctionWithSignature), "Something went wrong..."

        for epoch_idx in range(n_epochs):
            for batch_sample in dataloader:
                self.optimizer.zero_grad()
                self.substitute_parameters(**batch_sample)
                objective_value = objective(**batch_sample)
                objective_value.backward()
                self.optimizer.step()

            self.post_epoch(epoch_idx, objective_value.item())

    def define_problem(self):
        self.__is_problem_defined = True


class TorchProjectiveOptimizer:
    """Optimizer class that uses PyTorch as its optimization engine."""

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
        """Initialize an instance of TorchOptimizer.

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
        """Optimize the model with the given objective.

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


class BruteForceOptimizer:
    """Optimizer that searches for the optimal solution by evaluating all possible variants in parallel."""

    engine = "bruteforce"

    def __init__(self, possible_variants, N_parallel_processes=0):
        """Initialize an instance of BruteForceOptimizer.

        :param N_parallel_processes: number of processes to use in parallel
        :type N_parallel_processes: int
        :param possible_variants: list of possible variants to evaluate
        :type possible_variants: list
        """
        self.N_parallel_processes = N_parallel_processes
        self.possible_variants = possible_variants

    def element_wise_maximization(self, x):
        """Find the variant that maximizes the reward for a given element.

        :param x: element to optimize
        :type x: tuple
        :return: variant that maximizes the reward
        :rtype: int
        """

        def reward_function(variant):
            return self.objective(variant, x)

        reward_function = np.vectorize(reward_function)
        values = reward_function(self.possible_variants)
        return self.possible_variants[np.argmax(values)]

    def optimize(self, objective, weights):
        """Maximize the objective function over the possible variants.

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
