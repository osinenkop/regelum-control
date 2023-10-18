"""Contains main `Optimizable` class.
 
`Optimizable` class is to be used normally as a parent class for all objects that need to be optimized.
"""
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from regelum.__utilities import rc

try:
    from casadi import Opti

except (ModuleNotFoundError, ImportError):
    from unittest.mock import MagicMock

    Opti = MagicMock()


try:
    import torch

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
    NestedFunction,
)
from .core.hooks import requires_grad, detach, data_closure

import regelum


class Optimizable(regelum.RegelumBase):
    """Base class for all optimizable objects.

    This class is to be used normally as a parent class for all objects that need to be optimized.
    However, you can also use it as a separate instance and use all methods outside of the `Optimizable` class.
    """

    def __init__(self, optimizer_config: OptimizerConfig) -> None:
        """Initialize an optimizable object."""
        self.optimizer_config = optimizer_config
        self.kind = optimizer_config.kind

        self.__variables: VarContainer = VarContainer([])
        self.__functions: FuncContainer = FuncContainer(tuple())
        self.params_changed = False
        self.is_check_status = True
        self.__is_problem_defined = False

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
        self.opt_status = "unknown"

    @property
    def opt_options(self):
        return self.__opt_options

    @property
    def log_options(self):
        return self.__log_options

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

    def create_variable(
        self,
        *dims,
        name: str,
        is_constant=False,
        like=None,
        is_nested_function=False,
        nested_variables=None,
    ):
        metadata = None
        nested_variables = [] if nested_variables is None else nested_variables
        if not is_nested_function:
            metadata = self.create_variable_metadata(
                *dims, is_constant=is_constant, like=like
            )
            new_variable = OptimizationVariable(
                name=name, dims=dims, metadata=metadata, is_constant=is_constant
            )
        else:
            new_variable = NestedFunction(
                name=name,
                dims=dims,
                metadata=metadata,
                is_constant=is_constant,
                nested_variables=VarContainer(nested_variables),
            )
        if self.kind == "tensor" or self.kind == "numeric":
            if like is not None:
                new_variable = new_variable.with_data(like).with_metadata(like)

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
        act_on="all",
        discard_prev_sources=True,
        **source_kwargs,
    ):
        def source_hook(whatever):
            return func(
                source(),
                **{
                    kwarg: var() if isinstance(var, OptimizationVariable) else var
                    for kwarg, var in source_kwargs.items()
                },
            )

        def source_metadata_hook(whatever):
            return func(
                source(with_metadata=True),
                **{
                    kwarg: var(with_metadata=True)
                    if isinstance(var, OptimizationVariable)
                    else var
                    for kwarg, var in source_kwargs.items()
                },
            )

        data_hook = Hook(source_hook, act_on="data")
        metadata_hook = Hook(source_metadata_hook, act_on="metadata")

        if discard_prev_sources:
            for hook in connect_to.hooks:
                if "source" in hook.name:
                    connect_to.discard_hook(hook, disable_only=False)

        if act_on in ["all", "data"]:
            connect_to.register_hook(data_hook, first=True)
        if act_on in ["all", "metadata"]:
            connect_to.register_hook(metadata_hook, first=True)

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
        variable_min = rc.force_row(bounds[:, 0])
        variable_max = rc.force_row(bounds[:, 1])
        variable_initial_guess = (variable_min + variable_max) / 2
        if tile_parameter > 0:
            variable_sequence_initial_guess = rc.rep_mat(
                variable_initial_guess, 1, tile_parameter
            )
            sequence_min = rc.rep_mat(variable_min, tile_parameter, 1)
            sequence_max = rc.rep_mat(variable_max, tile_parameter, 1)
            result_bounds = rc.hstack((sequence_min, sequence_max))
            variable_initial_guess = variable_sequence_initial_guess
        else:
            result_bounds = bounds
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
            return bounds[:, : var.shape[1]] - var

        def ub_constr(var):
            return var - bounds[:, var.shape[1] :]

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
        constr = rc.vec(func.metadata) <= -1e-12
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
        if self.kind == "tensor":
            self.variables.substitute_data(**parameters)
            self.variables.substitute_metadata(**parameters)
        elif self.kind == "symbolic":
            params_to_delete = []
            for k, v in parameters.items():
                if k in self.variables:
                    if self.variables[k].is_constant:
                        self.__opti.set_value(self.variables[k](with_metadata=True), v)
                    else:
                        self.__opti.set_initial(
                            self.variables[k](with_metadata=True), v
                        )
                        params_to_delete.append(k)
            for k in params_to_delete:
                del parameters[k]
        elif self.kind == "numeric":
            self.variables.substitute_data(**parameters)
            self.variables.substitute_metadata(**parameters)

    def is_target_event(self, event):
        if self.__callback_target_events is None:
            return False

        return event in self.__callback_target_events

    def optimize_on_event(self, **parameters):
        raise NotImplementedError("optimize_on_event is not implemented")

    def optimize(self, raw=False, is_constrained=True, **parameters):
        if not self.__is_problem_defined:
            self.define_problem()

        result = None
        if self.kind == "symbolic":
            result = self.optimize_symbolic(**parameters, raw=raw)
        elif self.kind == "numeric":
            result = self.optimize_numeric(**parameters, raw=raw)
        elif self.kind == "tensor":
            self.optimize_tensor(**parameters, is_constrained=is_constrained)
            result = self.variables.decision_variables
        else:
            raise NotImplementedError

        return result

    @property
    def opt_func(self):
        return self.__opt_func

    @property
    def opti(self):
        return self.__opti

    def optimize_symbolic(self, raw=True, tol=1e-12, **kwargs):
        if self.__opt_func is None or self.params_changed:
            self.__opti.solver(
                self.opt_method, dict(self.__log_options), dict(self.__opt_options)
            )
            self.var_for_opti = VarContainer(
                [
                    variable
                    for variable in self.variables
                    if all(
                        [
                            "source" not in hook_name
                            for hook_name in variable.hooks.hooks_container.names
                        ]
                    )
                ]
            )
            self.__opt_func = self.__opti.to_function(
                "min_fun",
                [
                    variable(with_metadata=True)
                    for variable in self.var_for_opti.constants
                ],
                [
                    *[
                        variable(with_metadata=True)
                        for variable in self.var_for_opti.decision_variables
                    ],
                    self.objective.metadata,
                    *[func.metadata for func in self.constraints],
                ],
                list(self.var_for_opti.constants.names),
                [
                    *self.var_for_opti.decision_variables.names,
                    self.objective.name,
                    *[func.name for func in self.constraints],
                ],
            )
            self.params_changed = False

        self.substitute_parameters(**kwargs)

        result = self.__opt_func(
            **{
                k: v
                for k, v in kwargs.items()
                if k in self.var_for_opti.constants.names
            }
        )
        if self.is_check_status:
            self.update_status(result, tol=tol)
        return (
            result
            if raw
            else {
                name: value
                for name, value in result.items()
                if name in self.decision_variables.names
            }
        )

    def update_status(self, result=None, tol=1e-12):
        self.opt_status = "success"
        if self.kind == "symbolic":
            for constr_name in self.constraints.names:
                if rc.max(result[constr_name]) > tol:
                    self.opt_status = "failed"
                    break

    def optimize_numeric(self, raw=False, **parameters):
        self.substitute_parameters(**parameters)
        constraints = [
            NonlinearConstraint(func, -np.inf, 0) for func in self.constraints
        ]
        initial_guess = parameters.get(self.decision_variables.names[0])
        if initial_guess is None:
            assert (
                self.decision_variables[0].data is not None
            ), "Initial guess is None. Try to set 'like=...' when creating a decision variable"

            initial_guess = self.decision_variables[0].data

        opt_result = minimize(
            self.objective,
            x0=initial_guess,
            method=self.opt_method,
            bounds=self.__bounds if hasattr(self, "__bounds") else None,
            options=self.__opt_options,
            constraints=constraints,
            tol=1e-7,
        )
        return opt_result if raw else opt_result.x

    def instantiate_configuration(self):
        options = self.optimizer_config.config_options
        if self.optimizer is None:
            assert self.opt_method is not None and callable(
                self.opt_method
            ), f"Wrong optimization method {self.opt_method}."
            vars_to_opt = VarContainer(
                [
                    variable
                    for variable in self.decision_variables
                    if all(
                        [
                            "source" not in hook_name
                            for hook_name in variable.hooks.hooks_container.names
                        ]
                    )
                ]
            )
            assert len(vars_to_opt) == 1, "ambigous optimization variables"
            dvar = vars_to_opt[0]
            assert dvar is not None, "Couldn't find decision variable"
            assert isinstance(dvar, OptimizationVariable), "Something went wrong..."
            self.optimizer = self.opt_method(
                data_closure(dvar(with_metadata=False))(True), **self.__opt_options
            )
        n_epochs = options.get("n_epochs") if options.get("n_epochs") is not None else 1
        assert isinstance(n_epochs, int), "n_epochs must be an integer"
        assert len(self.functions.objectives) == 1, "Only one objective is supported"
        objective = self.functions.objectives[0]
        assert isinstance(objective, FunctionWithSignature), "Something went wrong..."

        return n_epochs, objective

    def eval_contraints(self):
        return sum(
            [
                rc.penalty_function(
                    func(**self.variables.to_data_dict()),
                    penalty_coeff=5,
                    delta=-3,
                )
                for func in self.constraints
            ]
        )

    def opt_constraints_tensor(self):
        n_epochs_per_constraint = 1
        if self.optimizer_config.config_options["constrained_optimization_policy"][
            "is_activated"
        ]:
            n_epochs_per_constraint = self.optimizer_config.config_options[
                "constrained_optimization_policy"
            ]["defaults"]["n_epochs_per_constraint"]
        for _ in range(n_epochs_per_constraint):
            self.optimizer.zero_grad()
            constr_value = self.eval_contraints()
            if (
                max(
                    [func(**self.variables.to_data_dict()) for func in self.constraints]
                ).item()
                < 0
            ):
                break
            constr_value.backward()
            self.optimizer.step()

        if (
            max(
                [func(**self.variables.to_data_dict()) for func in self.constraints]
            ).item()
            > 0
        ):
            self.opt_status = "failed"
        else:
            self.opt_status = "success"

    @apply_callbacks()
    def post_epoch(self, epoch_idx: int, objective_epoch_history: List[float]):
        return epoch_idx, objective_epoch_history

    def optimize_tensor_batch_sampler(
        self, batch_sampler, n_epochs, objective, is_constrained
    ):
        for epoch_idx in range(n_epochs):
            objective_epoch_history = []
            for batch_sample in batch_sampler:
                self.optimizer.zero_grad()
                self.substitute_parameters(**batch_sample)
                if is_constrained and len(self.constraints) > 0:
                    self.opt_constraints_tensor()
                    objective_value = objective(**self.variables.to_data_dict())
                    (objective_value + self.eval_contraints()).backward()
                else:
                    objective_value = objective(**self.variables.to_data_dict())
                    objective_value.backward()
                self.optimizer.step()
                objective_epoch_history.append(objective_value.item())
            if objective_value is not None:
                self.post_epoch(epoch_idx, objective_epoch_history)

        if (
            self.optimizer_config.config_options.get("is_reinstantiate_optimizer")
            is not None
        ):
            # Force to recreate torch.optim.Optimizer object before every optimization
            if self.optimizer_config.config_options.get("is_reinstantiate_optimizer"):
                self.optimizer = None

    def optimize_tensor(self, **parameters):
        n_epochs, objective = self.instantiate_configuration()
        batch_sampler = parameters.get("batch_sampler")
        is_constrained = parameters.get("is_constrained")
        if batch_sampler is not None:
            self.optimize_tensor_batch_sampler(
                batch_sampler=batch_sampler,
                n_epochs=n_epochs,
                objective=objective,
                is_constrained=is_constrained,
            )
        else:
            self.substitute_parameters(**parameters)
            for epoch_idx in range(n_epochs):
                self.optimizer.zero_grad()
                if is_constrained and len(self.constraints) > 0:
                    self.opt_constraints_tensor()
                    objective_value = objective(**self.variables.to_data_dict())
                    objective_value_constrained = +self.eval_contraints()
                    objective_value_constrained.backward()
                else:
                    objective_value = objective(**self.variables.to_data_dict())
                    objective_value.backward()
                self.optimizer.step()
                self.post_epoch(epoch_idx, [objective_value.item()])
            if self.optimizer_config.config_options.get("is_reinstantiate_optimizer"):
                self.optimizer = None

    def define_problem(self):
        self.__is_problem_defined = True

    def get_data_buffer_batch_size(self):
        config_options = self.optimizer_config.config_options

        method_name = config_options.get("data_buffer_sampling_method")
        kwargs = config_options.get("data_buffer_sampling_kwargs")
        assert (
            method_name is not None
        ), "Specify `data_buffer_sampling_method` in your optimizer_config"
        assert (
            kwargs is not None
        ), "Specify `data_buffer_sampling_kwargs` in your optimizer_config"

        if method_name == "iter_batches":
            return kwargs.get("batch_size")
        elif method_name == "sample_last":
            return kwargs.get("n_samples") if kwargs.get("n_samples") is not None else 1
        else:
            raise ValueError("Unknown data_buffer_sampling_method")
