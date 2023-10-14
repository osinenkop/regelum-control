"""Contains base classes of entities constitute the skeleton of the optimization procedure."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Union, Optional, Dict
from typing_extensions import Self
from inspect import signature
from collections.abc import Mapping
import warnings
from .hooks import get_data_hook


@dataclass
class OptimizationVariable:
    """Base class for all optimization variables.

    This is an object that represents a variable in the optimization procedure.
    It is a container for the data and metadata associated with the variable.
    Set the is_constant flag to True if the variable represents a constant.

    """

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
        # if isinstance(new_data, GeneratorType):
        #     new_data = list(new_data)

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
        """Construct a new OptimizationVariable object with the given metadata.

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

    def register_hook(self, hook: Union[Hook, Callable], first=False, act_on="all"):
        if not isinstance(hook, Hook):
            hook = Hook(hook, act_on=act_on)
        if (
            self.hooks.hooks_container is not None
            and hook.name in self.hooks.hooks_container.names
        ):
            if first:
                hook_to_delete = [
                    hook_tmp
                    for hook_tmp in self.hooks.hooks_container
                    if hook_tmp.name == hook.name
                ][0]
                self.hooks.remove_hook(hook_to_delete)
                self.register_hook(hook, first=True)
            else:
                warnings.warn(f"Hook {hook.name} already registered.", stacklevel=1)
        else:
            self.hooks.register_hook(hook, first=first)

    def enable_hook(self, hook):
        self.hooks.enable_hook(hook)

    def discard_hook(self, hook: FunctionWithSignature, disable_only=True):
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


@dataclass
class VarContainer(Mapping):
    """Container for optimization variables."""

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
        Tuple[OptimizationVariable],
        Tuple[OptimizationVariable, OptimizationVariable],
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
        hook: Optional[Hook] = None,
    ) -> None:
        """Hooks passed into this function are intended to be used here for torch grad setup.

        Hook setting grad is named requires_grad.
        Hook unset grad is named detach
        """
        for var in self.decision_variables.selected(variables_to_fix):
            var.as_constant(inplace=True)
            hook_names = [hook.name for hook in var.hooks]
            if "requires_grad" in hook_names:
                req_grad_hook = [
                    hook for hook in var.hooks if "requires_grad" in hook.name
                ][0]
                var.discard_hook(hook=req_grad_hook)
            if hook:
                assert (
                    hook.name == "detach"
                ), f"Unfixing of variable should be provided with detach hook, not {hook.name}"

                var.register_hook(hook=get_data_hook(var=var), first=True)
                var.register_hook(hook=hook, first=True)

    def unfix(
        self,
        variables_to_unfix: list[str],
        hook: Optional[Hook] = None,
    ) -> None:
        for var in self.constants.selected(variables_to_unfix):
            var.as_decision_variable(inplace=True)
            hook_names = [hook.name for hook in var.hooks]
            if "detach" in hook_names:
                detach_hook = [hook for hook in var.hooks if "detach" in hook.name][0]
                var.discard_hook(hook=detach_hook)
            if hook:
                assert (
                    hook.name == "requires_grad"
                ), f"Unfixing of variable should be provided with requires_grad hook, not {hook.name}"

                var.register_hook(hook=get_data_hook(var=var), first=True)
                var.register_hook(hook=hook, first=True)


@dataclass
class NestedFunction(OptimizationVariable):
    """A class representing nested functions.

    The variables which the nested function depends on are stored in the `nested_variables` field.
    """

    nested_variables: VarContainer = field(default_factory=lambda: VarContainer([]))

    def substitute_parameters(self, **parameters):
        self.nested_variables.substitute_data(**parameters)
        self.nested_variables.substitute_metadata(**parameters)

    def with_data(self, new_data, inplace=True):
        # if isinstance(new_data, GeneratorType):
        #     new_data = list(new_data)

        if inplace:
            self.data = new_data
            return self
        else:
            return NestedFunction(
                name=self.name,
                dims=self.dims,
                data=new_data,
                metadata=self.metadata,
                is_constant=self.is_constant,
                nested_variables=self.nested_variables,
            )

    def with_metadata(self, new_metadata, inplace=True):
        """Construct a new OptimizationVariable object with the given metadata.

        :param metadata: The metadata to associate with the variable.
        :type metadata: dict
        :return: A new OptimizationVariable object with the specified metadata.
        :rtype: OptimizationVariable
        """
        if inplace:
            self.metadata = new_metadata
            return self
        else:
            return NestedFunction(
                name=self.name,
                dims=self.dims,
                data=self.data,
                metadata=new_metadata,
                is_constant=self.is_constant,
                nested_variables=self.nested_variables,
            )


@dataclass
class FunctionWithSignature:
    """Wrapper class for functions, that parses a signature and ensures correctness of optimization procedure in the runtime."""

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
        return [var.name for var in self.variables.constants if var() is None]

    def __call__(
        self, *args, with_metadata: bool = False, raw_eval: bool = False, **kwargs
    ):
        """Call the function with the given keyword arguments.

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
            return self.func(*args, **kwargs_to_pass)

        if kwargs == {} and len(args) == 1:
            dvar = self.variables[self.free_placeholders[0]]
            is_decision_var_nested_function = isinstance(dvar, NestedFunction)
            if is_decision_var_nested_function:
                assert (
                    len(self.free_placeholders) == 1
                ), "The amount of free placeholders should be 1"
                dvar.substitute_parameters(
                    **{dvar.nested_variables.decision_variables.names[0]: args[0]}
                )
            return self.func(
                **{self.free_placeholders[0]: dvar()},
                **{
                    k: v
                    for k, v in self.variables.to_data_dict().items()
                    if k != self.free_placeholders[0]
                },
            )

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
        """Returns a list of free variables of the current function.

        Free variables are the arguments that
        are not defaulted and do not have a corresponding value.
        This method uses the signature of the function
        and the default parameters keys to determine the free variables.

        :return: A list of free variables of the current function.
        :rtype: tuple
        """
        signature_set = set(self.__signature)
        # default_keys = {
        #     name: data
        #     for name, data in self.variables.to_data_dict().items()
        #     if data is not None
        # }.keys()
        default_keys = set(self.variables.constants.names)
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
        hook: Optional[Self] = None,
    ):
        self.variables.fix(variables_to_fix, hook=hook)
        if data_dict:
            self.set_parameters(**data_dict)
        if metadata_dict:
            self.variables.substitute_metadata(**metadata_dict)

    def unfix_variables(
        self,
        variables_to_unfix: List[str],
        hook: Optional[Self] = None,
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

    def __add__(self, other) -> Optional[FuncContainer]:
        if isinstance(other, FunctionWithSignature):
            return FuncContainer((self, other))
        elif isinstance(other, FuncContainer):
            return FuncContainer((self, *other.functions))
        else:
            return None


@dataclass
class FuncContainer(Mapping):
    """A container for functions with signature."""

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


@dataclass
class Hook(FunctionWithSignature):
    """Base class for all hooks.

    Hooks are to be used with `OptimizationVariable` objects in order to modify `FunctionWithSignature` calling.
    """

    act_on: str = "all"


class ChainedHook:
    """A container for hooks."""

    def __init__(self, hooks: List[Hook]) -> None:
        """Initialize a chained hook object."""
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
                and self.enabled_hashmap[hook.name]
            ]
        )

    @property
    def data_hooks(self):
        return ChainedHook(
            [
                hook
                for hook in self.hooks_container
                if (hook.act_on == "data" or hook.act_on == "all")
                and self.enabled_hashmap[hook.name]
            ]
        )
