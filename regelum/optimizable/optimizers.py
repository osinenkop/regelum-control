"""Contains main `Optimizable` class.

`Optimizable` class is to be used normally as a parent class for all objects that need to be optimized.
"""

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from enum import Enum, auto

from regelum.utils import rg

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


class source_data_hook:
    def __init__(self, source, source_kwargs, func):
        self.source = source
        self.source_kwargs = source_kwargs
        self.func = func
        self.__name__ = "source_data_hook"

    def __call__(self, whatever):
        kwargs = {
            kwarg: var(True) if isinstance(var, OptimizationVariable) else var
            for kwarg, var in self.source_kwargs.items()
        }
        if self.source is None:
            return self.func(**kwargs)
        else:
            return self.func(self.source(True), **kwargs)


class source_metadata_hook:
    def __init__(self, source, source_kwargs, func):
        self.source = source
        self.source_kwargs = source_kwargs
        self.func = func
        self.__name__ = "source_metadata_hook"

    def __call__(self, whatever):
        kwargs = {
            kwarg: (
                var(with_metadata=True)
                if isinstance(var, OptimizationVariable)
                else var
            )
            for kwarg, var in self.source_kwargs.items()
        }
        if self.source is None:
            return self.func(**kwargs)
        else:
            return self.func(self.source(with_metadata=True), **kwargs)


class Optimizable(regelum.RegelumBase):
    """Base class for all optimizable objects.

    This class is to be used normally as a parent class for all objects that incapsulate optimization routines.
    However, you can also use it as a separate instance and use all methods outside of the `Optimizable` class.

    Args:
        optimizer_config (OptimizerConfig): An object of the
            OptimizerConfig class that defines the behavior of the
            optimizer.
    """

    def __init__(self, optimizer_config: OptimizerConfig) -> None:
        """Initialize an optimizable object.

        Args:
            optimizer_config (OptimizerConfig): An object of the
                OptimizerConfig class that defines the behavior of the
                optimizer.
        """
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
        """Get optimization options.

        Returns:
            dict: Optimization options
        """
        return self.__opt_options

    @property
    def log_options(self):
        """Get log options.

        Returns:
            dict: Log options
        """
        return self.__log_options

    @property
    def objective(self):
        """Get the objective.

        Returns:
            function/objective: Objective
        """
        assert len(self.objectives) == 1, "Ambiguous objective definition."
        return self.objectives[0]

    @property
    def objectives(self):
        """Get all objective functions.

        Returns:
            list: Objectives
        """
        return self.__functions.objectives

    @property
    def constraints(self):
        """Get all constraints.

        Returns:
            list: Constraints
        """
        return self.__functions.constraints

    @property
    def functions(self):
        """Get all functions.

        Returns:
            list: Functions
        """
        return self.__functions

    @property
    def variables(self) -> VarContainer:
        """Get all declared variables.

        Returns:
            VarContainer: Variables
        """
        return self.__variables

    def __recreate_opti(self):
        """Recreate optimization instance."""
        self.__opti = Opti()

    def __recreate_symbolic_variables(self):
        """Recreate symbolic variables in case if we use symbolic optimization and changed optimization procedure."""
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
        """Recreate symbolic functions in case if we use symbolic optimization and changed optimization procedure."""
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
        """Refresh all bound variables in the function list by updating their status."""
        for function in self.functions:
            for variable in function.variables:
                if isinstance(variable.data, OptimizationVariable):
                    variable.is_constant = variable.data.is_constant

    def __fix_variables_tensor(self, variables_to_fix, data_dict, metadata_dict):
        """Fix variables in case of self.kind == 'tensor' by setting them as constants.

        Args:
            variables_to_fix (List[str]): List of variable names to be
                fixed.
            data_dict (dict): Dictionary mapping variable names to their
                corresponding data.
            metadata_dict (dict): Dictionary mapping variable names to
                their corresponding metadata.
        """
        self.optimizer = None
        self.__variables.fix(variables_to_fix, hook=Hook(detach, act_on="data"))

    def __fix_variables_symbolic(self, variables_to_fix, data_dict, metadata_dict):
        """Fix variables in case of self.kind == 'symbolic' by setting them as constants.

        Args:
            variables_to_fix (List[str]): List of variable names to be
                fixed.
            data_dict (dict): Dictionary mapping variable names to their
                corresponding data.
            metadata_dict (dict): Dictionary mapping variable names to
                their corresponding metadata.
        """
        if metadata_dict is None:
            metadata_dict = {}
        passed_unfixed_variables = self.variables.selected(variables_to_fix)
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
        """Fix variables by setting them as constants.

        Args:
            variables_to_fix (List[str]): List of variable names to be
                fixed.
            data_dict (dict, optional): Optional dictionary mapping
                variable names to their corresponding data, defaults to
                None.
            metadata_dict (dict, optional): Optional dictionary mapping
                variable names to their corresponding metadata, defaults
                to None.
        """
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
        """Unfix tensor variables by setting them as non-constants.

        Args:
            variables_to_unfix (List[str]): List of variable names to be
                unfixed.
        """
        self.optimizer = None
        self.__variables.unfix(
            variables_to_unfix, hook=Hook(requires_grad, act_on="data")
        )

    def __unfix_variables_symbolic(self, variables_to_unfix):
        """Unfix symbolic variables by setting them as non-constants.

        Args:
            variables_to_unfix (List[str]): List of variable names to be
                unfixed.
        """
        passed_fixed_variables = self.variables.selected(variables_to_unfix)
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
        """Unfix variables by setting them as non-constants.

        Args:
            variables_to_unfix (List[str]): List of variable names to be
                unfixed.
        """
        if self.kind == "tensor":
            self.__unfix_variables_tensor(variables_to_unfix=variables_to_unfix)
        elif self.kind == "symbolic":
            self.__unfix_variables_symbolic(variables_to_unfix=variables_to_unfix)
        else:
            self.__variables.unfix(variables_to_unfix)

    def create_variable_metadata(self, *dims, is_constant=False, like=None):
        """Create metadata for a variable based on dimensions and characteristics.

        This method handles the creation of metadata for a variable which can either be
        a symbolic variable or a parameter, depending on the `is_constant` flag and
        whether the `like` parameter is used to clone an existing variable's properties.

        Args:
            *dims: Variable dimensions, can be a series of integers or
                tuples.
            is_constant: Flag indicating if the variable should be
                treated as a constant. Defaults to False.
            like: An existing variable whose shape and metadata should
                be replicated. Must have a `shape` attribute if
                provided.

        Returns:
            The created metadata object for the variable.

        Raises:
            AssertionError: If `like` doesn't have a `shape` attribute,
                or if the provided dimensions are invalid for symbolic
                variables.
        """
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
        """Create a new optimization variable or a nested function.

        Based on the provided dimensions and characteristics, this method either creates
        a standard optimization variable or a nested function variable. The newly created
        variable is then added to the internal variables container.

        Args:
            *dims: Dimensions of the variable.
            name: The name of the variable.
            is_constant: Specifies if the variable is a constant.
                Defaults to False.
            like: Optionally, an existing variable to base the new
                variable on.
            is_nested_function: Flag to determine if the variable is a
                nested function. Defaults to False.
            nested_variables: A list of variables that are nested within
                this variable, only used if `is_nested_function` is
                True.

        Returns:
            The newly created variable or nested function.
        """
        metadata = None
        nested_variables = [] if nested_variables is None else nested_variables
        if not is_nested_function:
            metadata = self.create_variable_metadata(
                *dims, is_constant=is_constant, like=like
            )
            new_variable = OptimizationVariable(
                name=name,
                dims=dims,
                metadata=metadata,
                data=like,
                is_constant=is_constant,
            )
        else:
            new_variable = NestedFunction(
                name=name,
                dims=dims,
                data=like,
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
        """Infer and register a symbolic function prototype using provided variables.

        This internal method uses the provided function signature and variable container
        to infer metadata and register a symbolic prototype of the function.

        Args:
            func: The function with a signature to infer metadata for.
            variables: A container of variables to use for the function.

        Returns:
            The function with inferred metadata.
        """
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
        """Register an objective function for optimization.

        The given function is wrapped with a function signature and is registered as an
        objective function within the internal functions container. It also ensures that
        the variables are properly contained within a VarContainer.

        Args:
            func: The objective function to be registered.
            variables: The variables over which the objective function
                is defined. Can be a single variable, a list of
                variables, or a VarContainer.

        Raises:
            AssertionError: If the new container for functions is None
                after attempting to register the objective.
        """
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
        source: Optional[OptimizationVariable] = None,
        act_on="all",
        discard_prev_sources=True,
        **source_kwargs,
    ):
        """Connect a source variable to another optimization variable, applying a given function.

        This method establishes a connection between a source optimization variable and the target
        optimization variable such that the specified function is applied to the source before
        being passed to the target. The function can take additional keyword arguments from
        `source_kwargs`.

        Args:
            connect_to: The optimization variable to which the source
                should be connected.
            func: The function to apply to the source variable before
                passing its value to `connect_to`.
            source: The source optimization variable, if any. If None,
                only `source_kwargs` are used.
            act_on: Specifies whether to act on "data", "metadata", or
                "all". Defaults to "all".
            discard_prev_sources: If True, previous sources connected to
                `connect_to` will be discarded. Defaults to True.
            **source_kwargs: Additional keyword arguments to pass to the
                function alongside the source variable.
        """
        data_hook = Hook(source_data_hook(source, source_kwargs, func), act_on="data")
        metadata_hook = Hook(
            source_metadata_hook(source, source_kwargs, func), act_on="metadata"
        )

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
        """Register a symbolic objective function into the optimization process.

        This internal method wraps the given function with additional metadata and sets it as the objective
        to minimize in the symbolic optimizer.

        Args:
            func: The objective function to be registered and minimized.
            variables: The container holding variables over which the
                objective function is defined.
        """
        func = self.__infer_and_register_symbolic_prototype(func, variables)
        self.__opti.minimize(func.metadata)

    @staticmethod
    def handle_bounds(
        bounds: Union[list, np.ndarray, None],
        dim_variable: int,
        tile_parameter: int = 0,
    ) -> Tuple:
        """Process bounds for optimization variables and prepares them for use in optimization.

        This static method handles the bounds for each dimension of an optimization variable,
        returning a tuple with the prepared bounds arrays necessary for the optimization process.

        Args:
            bounds: The bounds for each dimension of the variable.
                Defaults to (-inf, inf) if None.
            dim_variable: The number of dimensions of the optimization
                variable.
            tile_parameter: The number of times to tile the variable.
                Defaults to 0, meaning no tiling.

        Returns:
            A tuple containing the processed bounds, initial guess, and
            min-max ranges for the optimization variable.
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
        variable_min = rg.force_row(bounds[:, 0])
        variable_max = rg.force_row(bounds[:, 1])
        variable_initial_guess = (variable_min + variable_max) / 2
        if tile_parameter > 0:
            variable_sequence_initial_guess = rg.rep_mat(
                variable_initial_guess, 1, tile_parameter
            )
            sequence_min = rg.rep_mat(variable_min, tile_parameter, 1)
            sequence_max = rg.rep_mat(variable_max, tile_parameter, 1)
            result_bounds = rg.hstack((sequence_min, sequence_max))
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
        """Register a symbolic constraint into the optimization problem.

        This method infers the metadata for the provided functional constraint and formulates
        it as a symbolic constraint that the optimizer is subject to.

        Args:
            func: The constraint function with a signature.
            variables: The container of variables that the function
                operates on.
        """
        func = self.__infer_and_register_symbolic_prototype(func, variables)
        constr = rg.vec(func.metadata) <= -1e-8
        self.__opti.subject_to(constr)

    def __register_numeric_constraint(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        """Register a numeric constraint into the optimization problem.

        Currently, this method serves as a placeholder for future implementations where
        numerical constraints would be processed differently from symbolic ones.

        Args:
            func: The constraint function with a signature.
            variables: The container of variables that the function
                operates on.
        """
        func.declare_variables(variables)

    def __register_tensor_constraint(
        self, func: FunctionWithSignature, variables: VarContainer
    ):
        """Register a tensor constraint into the optimization problem.

        Currently, this method serves as a placeholder for future implementations where
        tensor constraints would be processed differently from symbolic ones.

        Args:
            func: The constraint function with a signature.
            variables: The container of variables that the function
                operates on.
        """
        func.declare_variables(variables)

    @property
    def constants(self):
        """Return the constants from the variables container.

        Returns:
            An object representing the constants in the optimization
            problem.
        """
        return self.__variables.constants

    @property
    def decision_variables(self):
        """Return the decision variables from the variables container.

        Returns:
            An object representing the decision variables in the
            optimization problem.
        """
        return self.__variables.decision_variables

    def substitute_parameters(self, **parameters):
        """Substitute parameters in the optimization problem with given values.

        This method is responsible for substituting the values of parameters (constants or
        initial guesses for decision variables) before optimization is carried out.

        Args:
            **parameters: Keyword arguments representing the parameter
                names and their values to substitute.
        """
        self.variables.substitute_data(**parameters)
        if self.kind == "symbolic":
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
        else:
            self.variables.substitute_metadata(**parameters)

    def is_target_event(self, event):
        """Check if an event is one of the target events for callback.

        Args:
            event: The event to check.

        Returns:
            True if the event is a target event, False otherwise.
        """
        if self.__callback_target_events is None:
            return False

        return event in self.__callback_target_events

    def optimize_on_event(self, **parameters):
        """Optimize the problem based on an event.

        This method is not implemented and serves as a placeholder for future functionality
        where optimization would be triggered by specific events.

        Raises:
            NotImplementedError: Always, as this method is not
                implemented.
        """
        raise NotImplementedError("optimize_on_event is not implemented")

    def optimize(self, raw=False, is_constrained=True, **parameters):
        """Optimize the problem based on the kind of optimizer.

        This method delegates the optimization to the appropriate method based on the optimizer's kind,
        whether it's symbolic (CasADi), numeric (SciPy), or tensor (Torch)-based.

        Args:
            raw: Determines if the raw optimization results should be
                returned. Defaults to False.
            is_constrained: Flags if the optimization includes
                constraints. Defaults to True.
            **parameters: Additional parameters for the optimization
                process.

        Returns:
            The result of the optimization process.

        Raises:
            NotImplementedError: If the optimizer kind is not
                recognized.
        """
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
        """Return the optimization function.

        Returns:
            The function used for optimization.
        """
        return self.__opt_func

    @property
    def opti(self):
        """A property that returns the current Opti instance.

        Returns:
            The Opti instance used for managing the optimization
            problem.
        """
        return self.__opti

    def optimize_symbolic(self, raw=True, tol=1e-12, **kwargs):
        """Optimize the symbolic (CasADi) problem with respect to the objective and constraints.

        This method prepares and executes the optimization process for a symbolic optimization
        problem. It generates a callable optimization function if it's not already defined or
        if the parameters have changed since the last optimization.

        Args:
            raw: If True, returns the raw optimization result as is. If
                False, filters the result to only include decision
                variables' names and their values. Defaults to True.
            tol: The tolerance for considering the optimization
                successful. Defaults to 1e-12.
            **kwargs: Additional keyword arguments used for parameter
                substitution before optimization.

        Returns:
            The optimization result. The structure depends on the `raw`
            parameter.
        """
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
            **{k: v for k, v in self.var_for_opti.constants.to_data_dict().items()}
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

    def update_status(self, result=None, tol=1e-8):
        """Update the optimization status based on the result.

        This method evaluates the result against a tolerance level to set the optimization status.
        If any constraint exceeds the tolerance, the status is set to failed.

        Args:
            result: The result of the optimization process.
            tol: The tolerance level used to evaluate constraints
                satisfaction. Defaults to 1e-8.
        """
        self.opt_status = OptStatus.success
        if self.kind == "symbolic":
            for constr_name in self.constraints.names:
                if rg.max(result[constr_name]) > tol:
                    self.opt_status = OptStatus.failed
                    break

    def optimize_numeric(self, raw=False, **parameters):
        """Optimize the numeric problem by using SciPy's minimize function.

        This method sets up and solves a numeric optimization problem. It substitutes parameters,
        defines constraints, and executes the optimization using the specified numerical method.

        Args:
            raw: If True, returns the raw `OptimizeResult` object from
                SciPy's minimize function. If False, returns the
                optimized decision variables as a NumPy array. Defaults
                to False.
            **parameters: Keyword arguments that include parameters to
                be substituted and any additional arguments for the
                optimization algorithm.

        Returns:
            The result of the numeric optimization. The structure
            depends on the `raw` parameter.

        Raises:
            AssertionError: If the initial guess is not provided and
                cannot be inferred from the decision variables.
        """
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
        """Instantiate the optimizer configuration for the optimization process.

        This method prepares the optimizer based on the decision variables and the optimization method specified.
        It also validates the configuration and ensures that a single decision variable and a single objective are present
        (In the current verion we consider optimization problem ambiguous if there are multiple decision variables or multiple objectives.
        If you want to optimize on several variables alternately, use `fix_variables` and `unfix_variables` explicitly).

        Returns:
            A tuple containing the number of epochs to run and the
            objective function.

        Raises:
            AssertionError: If the optimization method is not callable,
                if there are multiple or no decision variables, or if
                multiple objectives are present.
        """
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
        """Evaluate the constraints of the optimization problem.

        This method applies a penalty function to each constraint and sums up their values.

        Returns:
            The sum of the penalty functions applied to the constraints.
        """
        return sum(
            [
                rg.penalty_function(
                    func(**self.variables.to_data_dict()),
                    penalty_coeff=5,
                    delta=-3,
                )
                for func in self.constraints
            ]
        )

    def opt_constraints_tensor(self):
        """Perform constrained optimization for a tensor (Torch)-based optimizer.

        This method iterates over the constraints and performs gradient descent steps
        to minimize the constraints violations.

        Sets the optimization status to failed if any constraint is violated after optimization.
        """
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
            self.opt_status = OptStatus.failed
        else:
            self.opt_status = OptStatus.success

        print(constr_value)

    @apply_callbacks()
    def post_epoch(self, epoch_idx: int, objective_epoch_history: List[float]):
        """Emit post-epoch event.

        This a callback method called after each epoch during tensor-based optimization and is used to perform custom actions after each epoch, such as logging or saving epoch history.

        Args:
            epoch_idx: The index of the current epoch.
            objective_epoch_history: A list of objective function values
                from the current epoch.

        Returns:
            A tuple containing the current epoch index and the objective
            epoch history.
        """
        return epoch_idx, objective_epoch_history

    @apply_callbacks()
    def post_optimize(self):
        """Emit post-optimize event.

        This a callback method called after the completion of the optimization process and can be used to perform cleanup or finalization steps once optimization is finished.
        """
        pass

    def optimize_tensor_batch_sampler(
        self, batch_sampler, n_epochs, objective, is_constrained
    ):
        """Optimize the problem using a tensor (Torch)-based optimizer with batch sampling.

        This method iterates over the provided batch samples to perform epochs of optimization,
        optionally including constraint handling if specified.

        Args:
            batch_sampler: An iterable that provides batch samples for
                optimization.
            n_epochs: The number of epochs to run the optimization.
            objective: The objective function to be minimized.
            is_constrained: A boolean indicating whether to enforce
                constraints during optimization.
        """
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

        self.post_optimize()

    def define_problem(self):
        """Implement this method if you want to separate an optimization problem definition
        from the rest of your code. Needs only for readability.
        """
        self.__is_problem_defined = True

    def get_data_buffer_batch_size(self):
        """Retrieve the batch size for data sampling from the optimizer configuration.

        The method looks up the configuration for the data buffer sampling method and its associated
        keyword arguments to determine the batch size for sampling.

        Returns:
            The batch size determined by the data buffer sampling
            method.

        Raises:
            AssertionError: If the data_buffer_sampling_method or
                data_buffer_sampling_kwargs is not specified in the
                optimizer configuration.
            ValueError: If an unknown data_buffer_sampling_method is
                specified.
        """
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


class OptStatus(Enum):
    """Enum class representing optimization status."""

    success = auto()
    failed = auto()
