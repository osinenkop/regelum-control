# TODO: EXTEND DOCSTRING
"""
This module contains optimization routines 
to be used in optimal controllers, policies, critics etc.

"""
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from rcognita.__utilities import rc

try:
    from casadi import DM, MX, Function, Opti, nlpsol, vertcat

except (ModuleNotFoundError, ImportError):
    pass


try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    from rcognita.data_buffers import UpdatableSampler

except ModuleNotFoundError:
    from unittest.mock import MagicMock

    torch = MagicMock()
    UpdatableSampler = MagicMock()

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from inspect import signature
from typing import Any, Callable, Iterable, List, Tuple, Optional, Union, Dict
from typing_extensions import Self
from types import GeneratorType
import warnings

from rcognita.callbacks import apply_callbacks

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
        self.__dict__.update(**self.config_options)


class ChainedHook:
    def __init__(self, hooks: List["Hook"]):
        self.hooks = hooks
        self.hooks_container = sum(self.hooks) if len(hooks) > 0 else FuncContainer([])
        self.enabled_hashmap = {hook.name: True for hook in hooks}

    def __call__(self, arg, **kwargs):
        curr_result = arg
        for hook in [hook for hook in self.hooks if self.enabled_hashmap[hook.name]]:
            curr_result = self.call_hook(hook.name, curr_result, **kwargs)
        return curr_result

    def permute_order(self, hook1, hook2):
        for i, hook in enumerate(self.hooks):
            if hook == hook1:
                self.hooks[i] = hook2
            elif hook == hook2:
                self.hooks[i] = hook1

    def set_priority(self, hook1, hook2):
        assert (
            hook1 in self.hooks and hook2 in self.hooks
        ), f"Hook {hook1} or {hook2} not found."
        if self.hooks.index(hook1) > self.hooks.index(hook2):
            self.permute_order(hook1, hook2)

    def __iter__(self):
        for hook in self.hooks:
            yield hook

    def __len__(self):
        return len(self.hooks)

    def register_hook(self, hook, first=True):
        if hook not in self.hooks:
            self.__append_hook(hook, first=first)
        else:
            self.enabled_hashmap[hook.name] = True

    def enable_hook(self, hook):
        assert hook in self.hooks, f"Hook {hook} not found."
        self.enabled_hashmap[hook.name] = True

    def __append_hook(self, hook, first=True):
        if first:
            self.hooks.insert(0, hook)
        else:
            self.hooks.append(hook)

        self.hooks_container = (
            sum(self.hooks) if len(self.hooks) > 1 else FuncContainer(tuple(self.hooks))
        )
        self.enabled_hashmap[hook.name] = True

    def disable_hook(self, hook):
        assert hook in self.hooks, f"Hook {hook} not found."
        self.enabled_hashmap[hook.name] = False

    def remove_hook(self, hook):
        assert hook in self.hooks, f"Hook {hook} not found."
        self.hooks.remove(hook)
        del self.enabled_hashmap[hook.name]
        self.hooks_container = sum(self.hooks)

    def call_hook(self, name, *args, with_metadata=False, **kwargs):
        assert name in self.enabled_hashmap.keys(), f"Hook {name} not found."
        assert isinstance(
            self.hooks_container, FuncContainer
        ), f"Unexpected type of hook container: {type(self.hooks_container)}."
        hook = self.hooks_container[name]
        if isinstance(hook, FunctionWithSignature):
            return hook(*args, **kwargs, with_metadata=with_metadata)
        else:
            raise NotImplementedError

    @property
    def metadata_hooks(self):
        return ChainedHook(
            [
                hook
                for hook in self.hooks_container
                if (hook.act_on == "metadata" or hook.act_on == "all")
            ]
        )

    @property
    def data_hooks(self):
        return ChainedHook(
            [
                hook
                for hook in self.hooks_container
                if (hook.act_on == "data" or hook.act_on == "all")
            ]
        )


@dataclass(slots=True)
class OptimizationVariable:
    name: str
    dims: tuple
    data: Any = None
    metadata: Any = None
    is_constant: bool = False
    hooks: ChainedHook = field(default_factory=lambda: ChainedHook([]))

    def __call__(self, with_metadata: bool = False):
        obj_to_transform = self.data if not with_metadata else self.metadata
        hooks = self.hooks.metadata_hooks if with_metadata else self.hooks.data_hooks
        obj_to_transform = hooks(
            obj_to_transform, with_metadata=with_metadata, raw_eval=True
        )

        return obj_to_transform

    def renamed(self, new_name: str, inplace=True) -> Self:
        if inplace:
            self.name = new_name
            return self
        else:
            return OptimizationVariable(
                name=new_name,
                dims=self.dims,
                data=self.data,
                metadata=self.metadata,
                is_constant=self.is_constant,
            )

    def with_data(self, new_data, inplace=True):
        if isinstance(new_data, GeneratorType):
            new_data = list(new_data)

        if inplace:
            self.data = new_data
            return self
        else:
            return OptimizationVariable(
                name=self.name,
                dims=self.dims,
                data=new_data,
                metadata=self.metadata,
                is_constant=self.is_constant,
            )

    def with_metadata(self, new_metadata, inplace=True):
        """
        Constructs a new OptimizationVariable object with the given metadata.

        :param metadata: The metadata to associate with the variable.
        :type metadata: dict
        :return: A new OptimizationVariable object with the specified metadata.
        :rtype: OptimizationVariable
        """
        if inplace:
            self.metadata = new_metadata
            return self
        else:
            return OptimizationVariable(
                name=self.name,
                dims=self.dims,
                data=self.data,
                metadata=new_metadata,
                is_constant=self.is_constant,
            )

    def as_constant(self, inplace=True):
        if self.is_constant:
            return self
        else:
            if inplace:
                self.is_constant = True
                return self
            else:
                return OptimizationVariable(
                    name=self.name,
                    dims=self.dims,
                    data=self.data,
                    metadata=self.metadata,
                    is_constant=True,
                )

    def as_decision_variable(self, inplace=True):
        if not self.is_constant:
            return self
        else:
            if inplace:
                self.is_constant = False
                return self
            return OptimizationVariable(
                name=self.name,
                dims=self.dims,
                data=self.data,
                metadata=self.metadata,
                is_constant=False,
            )

    def as_metadata_dict(self):
        return {self.name: self.metadata}

    def as_data_dict(self):
        return {self.name: self.data}

    def as_dims_dict(self):
        return {self.name: self.dims}

    def __radd__(self, other):
        return VarContainer((self,))

    def __add__(self, other):
        if isinstance(other, OptimizationVariable):
            return VarContainer([self, other])
        elif isinstance(other, VarContainer):
            return VarContainer((self,) + tuple(other.variables))

    def register_hook(self, hook: Union["Hook", Callable], first=False, act_on="all"):
        if not isinstance(hook, Hook):
            hook = Hook(hook, act_on=act_on)
        if (
            self.hooks.hooks_container is not None
            and hook.name in self.hooks.hooks_container.names
        ):
            warnings.warn(f"Hook {hook.name} already registered.")
        else:
            self.hooks.register_hook(hook, first=first)

    def enable_hook(self, hook):
        self.hooks.enable_hook(hook)

    def discard_hook(self, hook: "FunctionWithSignature", disable_only=True):
        if disable_only:
            self.hooks.disable_hook(hook)
        else:
            self.hooks.remove_hook(hook)

    def __str__(self):
        return (
            f"{self.name}\n  "
            + f"data: {self()}\n  "
            + f"metadata: {self(with_metadata=True)}\n  "
            + f"dims: {self.dims}\n  "
            + f"is_constant: {self.is_constant}\n\n"
        )


@dataclass(slots=True)
class VarContainer(Mapping):
    _variables: Union[
        list[OptimizationVariable],
        Tuple[OptimizationVariable],
        Tuple[OptimizationVariable, OptimizationVariable],
    ]

    def __post_init__(self):
        self._variables = tuple(self._variables)

    @property
    def variables_hashmap(self):
        return {var.name: var for var in self._variables}

    def to_dict(self):
        return self.variables_hashmap

    def __repr__(self):
        return "VarContainer:\n  " + "".join(
            [f"{variable}\n  " for variable in self._variables]
        )

    def __str__(self):
        return "VarContainer:\n  " + "".join(
            [f"{variable}\n  " for variable in self._variables]
        )

    @property
    def metadatas(self):
        return tuple(var.metadata for var in self.variables)

    def to_data_dict(self):
        return {var.name: var() for var in self.variables}

    def to_metadata_dict(self):
        return {var.name: var(with_metadata=True) for var in self.variables}

    def selected(self, var_names: List[str]) -> Tuple[OptimizationVariable]:
        return tuple(var for var in self.variables if var.name in var_names)

    def substitute_data(self, **name_data_dict) -> Self:
        for var in self.selected(list(name_data_dict.keys())):
            new_data = name_data_dict.get(var.name)
            var.with_data(new_data, inplace=True)
        return self

    def substitute_metadata(self, **name_metadata_dict) -> Self:
        for var in self.selected(list(name_metadata_dict.keys())):
            new_metadata = name_metadata_dict.get(var.name)
            var.with_metadata(new_metadata, inplace=True)
        return self

    @property
    def variables(
        self,
    ) -> Union[
        Tuple[OptimizationVariable], Tuple[OptimizationVariable, OptimizationVariable]
    ]:
        if isinstance(self._variables, tuple):
            return self._variables
        elif isinstance(self._variables, list):
            return tuple(self._variables)
        else:
            raise TypeError("Something went wrong with the type of the variables.")

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

    @property
    def names(self):
        return tuple(var.name for var in self.variables)

    def __radd__(self, other) -> Self:
        return self

    def __add__(self, other) -> Self:
        if (
            isinstance(other, VarContainer)
            and isinstance(other.variables, tuple)
            and isinstance(self.variables, tuple)
        ):
            return VarContainer(other.variables + self.variables)
        elif isinstance(other, OptimizationVariable) and isinstance(
            self.variables, tuple
        ):
            return VarContainer(self.variables + (other,))
        elif other is None:
            return self
        else:
            raise NotImplementedError

    def __iter__(self):
        for variable in self.variables:
            yield variable

    def __len__(self):
        return len(self._variables)

    def __getitem__(self, key) -> Union[OptimizationVariable, Self]:
        if isinstance(key, int) and isinstance(self.variables, tuple):
            assert 0 <= key < len(self.variables), f"Index {key} is out of bounds."
            return self.variables[key]
        elif isinstance(key, slice) and isinstance(self.variables, tuple):
            return VarContainer(self.variables[key])
        elif isinstance(key, str) and self.variables_hashmap is not None:
            res = self.variables_hashmap.get(key)
            assert res is not None, f"OptimizationVariable {key} not found."
            return res
        else:
            raise NotImplementedError

    def fix(
        self,
        variables_to_fix: list[str],
        hook: Optional["Hook"] = None,
    ) -> None:
        """
        Hooks passed into this function are intended to be used here for torch grad setup.
        Hook setting grad is named _requires_grad.
        Hook unset grad is named _detach
        """
        for var in self.decision_variables.selected(variables_to_fix):
            var.as_constant(inplace=True)
            hook_names = [hook.name for hook in var.hooks]
            if "_requires_grad" in hook_names:
                req_grad_hook = [
                    hook for hook in var.hooks if "_requires_grad" in hook.name
                ][0]
                var.discard_hook(hook=req_grad_hook)
            if hook:
                assert (
                    hook.name == "_detach"
                ), f"Unfixing of variable should be provided with _detach hook, not {hook.name}"
                var.register_hook(hook=hook, first=True)

    def unfix(
        self,
        variables_to_unfix: list[str],
        hook: Optional["Hook"] = None,
    ) -> None:
        for var in self.constants.selected(variables_to_unfix):
            var.as_decision_variable(inplace=True)
            hook_names = [hook.name for hook in var.hooks]
            if "_detach" in hook_names:
                detach_hook = [hook for hook in var.hooks if "_detach" in hook.name][0]
                var.discard_hook(hook=detach_hook)
            if hook:
                assert (
                    hook.name == "_requires_grad"
                ), f"Unfixing of variable should be provided with _requires_grad hook, not {hook.name}"
                var.register_hook(hook=hook, first=True)


@dataclass
class FunctionWithSignature:
    func: Callable
    variables: VarContainer = field(default_factory=lambda: VarContainer([]))
    is_objective: bool = False
    metadata: Any = None

    def __post_init__(self) -> None:
        self.__signature = self.__parse_signature(self.func)
        parameter_names = set(self.variables.names)
        kwargs_intersection = parameter_names & set(self.__signature)
        if kwargs_intersection != parameter_names:
            raise ValueError(
                "Unknown variables encountered: "
                f"{self.variables.keys() - kwargs_intersection}"
            )

        self.name = self.func.__name__

    @property
    def await_constants(self):
        return not (self.constants_to_substitute == [])

    @property
    def constants_to_substitute(self):
        return [var.name for var in self.variables.constants if var.data is None]

    def __call__(
        self, *args, with_metadata: bool = False, raw_eval: bool = False, **kwargs
    ):
        """
        Call the function with the given keyword arguments.
        Only keyword arguments that are set will be passed to the function.

        :param kwargs: The keyword arguments to be passed to the function.
        :type kwargs: dict
        :return: The return value of the function.
        :raises ValueError: If not all required parameters have been set.
        :rtype: Any
        """
        if raw_eval:
            kwargs_to_pass = {
                k: v for k, v in kwargs.items() if k in self.variables.names
            }
            return self.func(*args, **kwargs)
        if kwargs == {} and len(args) == 1:
            return self.func(**{self.free_placeholders[0]: args[0]})

        kwargs_to_pass = {
            k: v for k, v in kwargs.items() if k in self.free_placeholders
        }

        if with_metadata:
            return self.func(
                **{**kwargs_to_pass, **self.variables.constants.to_metadata_dict()}
            )
        elif not self.await_constants:
            return self.func(
                **{
                    **kwargs_to_pass,
                    **self.variables.constants.to_data_dict(),
                }
            )
        else:
            raise ValueError(
                f"Not all declared constants were substituted: {self.constants_to_substitute}"
            )

    @property
    def signature(self) -> tuple:
        return tuple(self.__signature)

    @property
    def occupied(self) -> tuple:
        return tuple(self.variables.constants.names)

    @property
    def free_placeholders(self) -> tuple:
        """
        Returns a list of free variables of the current function.
        Free variables are the arguments that
        are not defaulted and do not have a corresponding value.
        This method uses the signature of the function
        and the default parameters keys to determine the free variables.

        :return: A list of free variables of the current function.
        :rtype: list
        """

        signature_set = set(self.__signature)
        default_keys = {
            name: data
            for name, data in self.variables.to_data_dict().items()
            if data is not None
        }.keys()
        return tuple(signature_set - default_keys)

    def declare_variables(
        self,
        variables: Union[VarContainer, OptimizationVariable],
        replace=False,
    ) -> Self:
        if isinstance(variables, OptimizationVariable):
            variables = VarContainer([variables])
        unknown_params = [
            name for name in variables.names if name not in self.__signature
        ]
        assert unknown_params == [], f"Unknown parameters: {unknown_params}"

        if not replace:
            if self.variables is not None:
                new_variables = self.variables + variables
                if new_variables is not None:
                    self.variables = self.variables + variables
        else:
            self.variables = variables
        return self

    def set_parameters(self, **kwargs) -> None:
        kwargs_intersection = kwargs.keys() & self.occupied
        if kwargs_intersection != kwargs.keys():
            raise ValueError(
                f"Unknown parameters encountered: {kwargs.keys() - kwargs_intersection}"
            )
        self.variables.substitute_data(**kwargs)

    def fix_variables(
        self,
        variables_to_fix: List[str],
        data_dict: Optional[Dict],
        metadata_dict: Optional[Dict],
        hook: Optional["FunctionWithSignature"] = None,
    ):
        self.variables.fix(variables_to_fix, hook=hook)
        if data_dict:
            self.set_parameters(**data_dict)
        if metadata_dict:
            self.variables.substitute_metadata(**metadata_dict)

    def unfix_variables(
        self,
        variables_to_unfix: List[str],
        hook: Optional["FunctionWithSignature"] = None,
    ):
        self.variables.unfix(variables_to_unfix, hook=hook)

    def __parse_signature(self, func: Callable) -> Tuple[str]:
        signature_list = []

        variables = (
            func.signature
            if hasattr(func, "signature")
            else signature(func).parameters.values()
        )
        for param in variables:
            if param.kind == param.VAR_POSITIONAL:
                raise ValueError("Undefined number of arguments")
            if param.kind == param.VAR_KEYWORD:
                raise ValueError("Undefined number of keyword arguments")
            signature_list.append(param.name)

        return tuple(signature_list)

    def __radd__(self, other):
        return FuncContainer((self,))

    def __add__(self, other) -> Optional["FuncContainer"]:
        if isinstance(other, FunctionWithSignature):
            return FuncContainer((self, other))
        elif isinstance(other, FuncContainer):
            return FuncContainer((self, *other.functions))
        else:
            return None


@dataclass(slots=True)
class FuncContainer(Mapping):
    _functions: Union[
        Tuple[FunctionWithSignature, FunctionWithSignature],
        Tuple[FunctionWithSignature],
    ]

    def __post_init__(self):
        self._functions = tuple(self._functions)

    @property
    def functions_hashmap(self):
        return {func.name: func for func in self._functions}

    def to_dict(self):
        return self.functions_hashmap

    def __repr__(self):
        return "FuncContainer:\n  " + "".join(
            [f"{function}\n  " for function in self._functions]
        )

    def __str__(self):
        return "FuncContainer:\n  " + "".join(
            [f"{function}\n  " for function in self._functions]
        )

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
        return len(self.functions)

    def __getitem__(self, key) -> Union[FunctionWithSignature, Self]:
        if isinstance(key, int):
            assert 0 <= key < len(self.functions), f"Index {key} is out of bounds."
            return self.functions[key]
        elif isinstance(key, slice):
            return FuncContainer(self.functions[key])
        elif isinstance(key, str) and self.functions_hashmap is not None:
            res = self.functions_hashmap.get(key)
            assert res is not None, f"Function {key} not found."
            return res
        else:
            raise NotImplementedError

    @property
    def names(self):
        return tuple(function.name for function in self.functions)


@dataclass(slots=True)
class Hook(FunctionWithSignature):
    act_on: str = "all"


class Optimizable(RcognitaBase):
    def __init__(self, optimizer_config: OptimizerConfig) -> None:
        self.optimizer_config = optimizer_config
        self.kind = optimizer_config.kind
        self.__is_problem_defined = False
        self.__variables: VarContainer = VarContainer([])
        self.__functions: FuncContainer = FuncContainer(tuple())
        self.objective_name = None

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
    def objective(self):
        assert (
            len(self.objectives) == 1 or self.objective_name is not None
        ), "Ambiguous objective definition. Specify objective name."
        return self.objectives[self.objective_name]

    def _requires_grad(self, variable_data):
        if self.kind == "tensor":
            if isinstance(variable_data, List):
                for datum in variable_data:
                    datum[1].requires_grad_(True)
            else:
                variable_data[1].requires_grad_(True)
        return variable_data

    def _detach(self, variable_data):
        if self.kind == "tensor":
            if isinstance(variable_data, List):
                for datum in variable_data:
                    datum[1].requires_grad_(False)
            else:
                variable_data[1].requires_grad_(False)
        return variable_data

    def _mutate_metadata(self, new_metadata, tag="default"):
        def metadata_mutator(whatever):
            return new_metadata

        return Hook(metadata_mutator, metadata=tag, act_on="metadata")

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

    def fix_variables(
        self,
        variables_to_fix: List[str],
        data_dict: Optional[Dict] = None,
        metadata_dict: Optional[Dict] = None,
    ):
        if self.kind == "tensor":
            self.__variables.fix(
                variables_to_fix, hook=Hook(self._detach, act_on="data")
            )
        elif self.kind == "symbolic":
            if metadata_dict is None:
                metadata_dict = {}
            passed_unfixed_variables = sum(self.variables.selected(variables_to_fix))
            assert isinstance(
                passed_unfixed_variables, VarContainer
            ), "An error occured while fixing variables."

            for variable in passed_unfixed_variables:
                if "metadata_mutator" not in variable.hooks.hooks_container.names:
                    metadata = self.create_variable_metadata(
                        *variable.dims, is_constant=True
                    )
                    metadata_dict[variable.name] = metadata
                    variable.register_hook(
                        self._mutate_metadata(metadata, tag="fix"), first=True
                    )
                else:
                    hook = variable.hooks.hooks_container["metadata_mutator"]
                    if hook.metadata == "unfix":
                        variable.discard_hook(hook)
                    elif hook.metadata == "fix":
                        variable.enable_hook(hook)

            self.__variables.fix(variables_to_fix)
            # self.__variables.substitute_metadata(**metadata_dict)
            for function in self.functions:
                metafunc = function(**self.variables.to_metadata_dict(), raw_eval=True)
                function.metadata = metafunc
                if function.is_objective:
                    self.__opti.minimize(metafunc)
                else:
                    self.__opti.subject_to(metafunc < 0)
        else:
            self.__variables.fix(variables_to_fix)

    def unfix_variables(
        self,
        variables_to_unfix: List[str],
    ):
        if self.kind == "tensor":
            self.__variables.unfix(
                variables_to_unfix, hook=Hook(self._requires_grad, act_on="data")
            )
        elif self.kind == "symbolic":
            metadata_dict = {}
            passed_fixed_variables = sum(self.variables.selected(variables_to_unfix))
            assert isinstance(
                passed_fixed_variables, VarContainer
            ), "An error occured while fixing variables."

            for variable in passed_fixed_variables:
                if "metadata_mutator" not in variable.hooks.hooks_container.names:
                    metadata = self.create_variable_metadata(
                        *variable.dims, is_constant=False
                    )
                    metadata_dict[variable.name] = metadata
                    variable.register_hook(
                        self._mutate_metadata(metadata, tag="unfix"), first=True
                    )
                else:
                    hook = variable.hooks.hooks_container["metadata_mutator"]
                    if hook.metadata == "fix":
                        variable.discard_hook(hook)
                    elif hook.metadata == "unfix":
                        variable.enable_hook(hook)

            self.__variables.fix(variables_to_unfix)
            # self.__variables.substitute_metadata(**metadata_dict)
            for function in self.functions:
                metafunc = function(**self.variables.to_metadata_dict(), raw_eval=True)
                function.metadata = metafunc
                if function.is_objective:
                    self.__opti.minimize(metafunc)
                else:
                    self.__opti.subject_to(metafunc < 0)

        else:
            self.__variables.unfix(variables_to_unfix)

    def create_variable_metadata(self, *dims, is_constant=False):
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
        return metadata

    def create_variable(self, *dims, name: str, is_constant=False):
        metadata = self.create_variable_metadata(*dims, is_constant=is_constant)
        new_variable = OptimizationVariable(
            name=name, dims=dims, metadata=metadata, is_constant=is_constant
        )
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
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
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

    def register_constraint(
        self, func: Callable, variables: List[OptimizationVariable]
    ):
        func = FunctionWithSignature(func)
        _variables = VarContainer(variables)

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
        # ToDo: add multiple objectives
        if self.__opt_func is None:
            self.__opti.solver(self.opt_method, self.__log_options, self.__opt_options)
            self.__opt_func = self.__opti.to_function(
                "min_fun",
                list(self.variables.metadatas),
                [*self.decision_variables.metadatas, self.objective],
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
            self.objective,
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
        for _ in range(n_epochs):
            for batch_sample in dataloader:
                self.optimizer.zero_grad()
                objective_value = objective(batch_sample)
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
        self.sheduler = (
            self.sheduler_method(self.optimizer, **self.sheduler_options)
            if self.sheduler_method is not None
            else None
        )

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
        if self.sheduler:
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

        def reward_function(variant):
            return self.objective(variant, x)

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
