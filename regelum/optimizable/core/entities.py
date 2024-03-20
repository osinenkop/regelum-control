"""The `entities` module of the optimization framework defines the core entities involved in setting up and managing the optimization problem.

This module contains the following key classes:

- `OptimizationVariable`: Represents a variable within the optimization process, which can be marked as a constant or a decision variable.
- `VarContainer`: A container class that holds and manages multiple `OptimizationVariable` instances, providing utility methods for variable manipulation.
- `NestedFunction`: A specialized form of `OptimizationVariable` that represents a nested function with its own set of dependent variables.
- `FunctionWithSignature`: A wrapper for functions within the optimization context, ensuring that the function signature matches the expected variables and managing constants and placeholders.
- `FuncContainer`: A container class that holds and manages multiple `FunctionWithSignature` instances, providing utility methods for function manipulation.
- `Hook`: A class representing a hook that can be applied to an `OptimizationVariable`.
- `ChainedHook`: A class representing a chain of hooks that can be applied to an `OptimizationVariable` one-by-one in a specific order.
These entities are designed to work together to define and manipulate the variables and functions that form the basis of an optimization problem or change it at runtime.
"""

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
    """Represent an optimization variable within the optimization process.

    This base class encapsulates data and metadata for variables used in optimization,
    allowing them to be treated as either constants or decision variables.

    Args:
        name: The name of the variable.
        dims: The dimensions of the variable as a tuple.
        data: The initial data for the variable. Defaults to None.
        metadata: The metadata associated with the variable. Defaults to
            None.
        is_constant: Flag indicating if the variable is a constant.
            Defaults to False.
        hooks: A ChainedHook instance containing hooks to be applied to
            the variable. Defaults to an empty ChainedHook.
    """

    name: str
    dims: tuple
    data: Any = None
    metadata: Any = None
    is_constant: bool = False
    hooks: ChainedHook = field(default_factory=lambda: ChainedHook([]))

    def __call__(self, with_metadata: bool = False):
        """Retrieve the data or metadata of the variable after applying hooks.

        Args:
            with_metadata: If True, retrieve metadata; otherwise,
                retrieve data. Defaults to False.

        Returns:
            The data or metadata of the variable after hooks
            transformation.
        """
        obj_to_transform = self.data if not with_metadata else self.metadata
        hooks = self.hooks.metadata_hooks if with_metadata else self.hooks.data_hooks
        obj_to_transform = hooks(
            obj_to_transform, with_metadata=with_metadata, raw_eval=True
        )

        return obj_to_transform

    def renamed(self, new_name: str, inplace=True) -> Self:
        """Rename the optimization variable.

        Args:
            new_name: The new name for the variable.
            inplace: If True, modify the variable in place; otherwise,
                return a new instance. Defaults to True.

        Returns:
            The optimization variable with the updated name (self or new
            instance).
        """
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
        """Associate new data with the optimization variable.

        Args:
            new_data: The new data to associate with the variable.
            inplace: If True, modify the variable in place; otherwise,
                return a new instance. Defaults to True.

        Returns:
            The optimization variable with the updated data (self or new
            instance).
        """
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
        """Associate new metadata with the optimization variable.

        Args:
            new_metadata: The new metadata to associate with the
                variable.
            inplace: If True, modify the variable in place; otherwise,
                return a new instance. Defaults to True.

        Returns:
            The optimization variable with the updated metadata (self or
            new instance).
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
        """Mark the variable as a constant.

        Args:
            inplace: If True, modify the variable in place; otherwise,
                return a new instance marked as constant. Defaults to
                True.

        Returns:
            The optimization variable marked as a constant (self or new
            instance).
        """
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
        """Mark the variable as a decision variable.

        Args:
            inplace: If True, modify the variable in place; otherwise,
                return a new instance marked as a decision variable.
                Defaults to True.

        Returns:
            The optimization variable marked as a decision variable
            (self or new instance).
        """
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
        """Convert the variable's metadata to a dictionary.

        Returns:
            A dictionary with the variable's name as the key and its
            metadata as the value.
        """
        return {self.name: self.metadata}

    def as_data_dict(self):
        """Convert the variable's data to a dictionary.

        Returns:
            A dictionary with the variable's name as the key and its
            data as the value.
        """
        return {self.name: self.data}

    def as_dims_dict(self):
        """Convert the variable's dimensions to a dictionary.

        Returns:
            A dictionary with the variable's name as the key and its
            dimensionsas the value.
        """
        return {self.name: self.dims}

    def __radd__(self, other):
        """Support the addition of this variable to another object by returning a VarContainer.

        Args:
            other: The object to add this variable to.

        Returns:
            A VarContainer containing this variable.
        """
        return VarContainer((self,))

    def __add__(self, other):
        """Add another variable or container to this variable to form a VarContainer.

        Args:
            other: The variable or VarContainer to be added.

        Returns:
            A VarContainer including this variable and `other`.
        """
        if isinstance(other, OptimizationVariable):
            return VarContainer([self, other])
        elif isinstance(other, VarContainer):
            return VarContainer((self,) + tuple(other.variables))

    def register_hook(self, hook: Union[Hook, Callable], first=False, act_on="all"):
        """Register a hook to the optimization variable.

        Args:
            hook: The hook or callable to be added as a hook.
            first: If True, the hook will be added at the beginning.
                Defaults to False.
            act_on: Specifies the target of the hook, either "data" or
                "metadata" or "all". Defaults to "all".
        """
        if not isinstance(hook, Hook):
            hook = Hook(hook, act_on=act_on)
        if self.hooks.hooks_container and hook.name in self.hooks.hooks_container.names:
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
        """Enable a previously added hook.

        Args:
            hook: The hook to be enabled.
        """
        self.hooks.enable_hook(hook)

    def discard_hook(self, hook: FunctionWithSignature, disable_only=True):
        """Discard or disable a hook from the optimization variable.

        Args:
            hook: The hook to be discarded or disabled.
            disable_only: If True, the hook will only be disabled, not
                removed. Defaults to True.
        """
        if disable_only:
            self.hooks.disable_hook(hook)
        else:
            self.hooks.remove_hook(hook)

    def __str__(self):
        """Return a string representation of the optimization variable.

        Returns:
            A formatted string with the variable's details.
        """
        return (
            f"{self.name}\n  "
            + f"data: {self()}\n  "
            + f"metadata: {self(with_metadata=True)}\n  "
            + f"dims: {self.dims}\n  "
            + f"is_constant: {self.is_constant}\n\n"
        )


@dataclass
class VarContainer(Mapping):
    """A container class for managing multiple optimization variables.

    Args:
        _variables: A collection of OptimizationVariables to be
            contained.
    """

    _variables: Union[
        list[OptimizationVariable],
        Tuple[OptimizationVariable],
        Tuple[OptimizationVariable, OptimizationVariable],
    ]

    def __post_init__(self):
        """Initialize the VarContainer by ensuring variables are stored as a tuple."""
        self._variables = tuple(self._variables)

    @property
    def variables_hashmap(self):
        """Create a hashmap from variable names to variables.

        Returns:
            A dictionary mapping variable names to their respective
            OptimizationVariable objects.
        """
        return {var.name: var for var in self._variables}

    def to_dict(self):
        """Convert the container to a dictionary.

        Returns:
            A dictionary representation of the VarContainer.
        """
        return self.variables_hashmap

    def __repr__(self):
        """Return a string representation of the VarContainer for debugging.

        Returns:
            A debug string of the VarContainer.
        """
        return "VarContainer:\n  " + "".join(
            [f"{variable}\n  " for variable in self._variables]
        )

    def __str__(self):
        """Return a string representation of the VarContainer.

        Returns:
            A formatted string of the VarContainer.
        """
        return "VarContainer:\n  " + "".join(
            [f"{variable}\n  " for variable in self._variables]
        )

    @property
    def metadatas(self):
        """Retrieve the metadata of all contained variables.

        Returns:
            A tuple with the metadata of each OptimizationVariable.
        """
        return tuple(var.metadata for var in self.variables)

    def to_data_dict(self):
        """Convert variable data to a dictionary.

        Returns:
            A dictionary with variable names as keys and their data as
            values.
        """
        return {var.name: var() for var in self.variables}

    def to_metadata_dict(self):
        """Convert variable metadata to a dictionary.

        Returns:
            A dictionary with variable names as keys and their metadata
            as values.
        """
        return {var.name: var(with_metadata=True) for var in self.variables}

    def selected(self, var_names: List[str]) -> Tuple[OptimizationVariable]:
        """Select a subset of variables by names.

        Args:
            var_names: A list of names of the variables to select.

        Returns:
            A VarContainer containing only the selected variables.
        """
        return VarContainer([var for var in self.variables if var.name in var_names])

    def substitute_data(self, **name_data_dict) -> Self:
        """Substitute data for selected variables.

        Args:
            **name_data_dict: A dictionary with variable names as keys
                and new data as values.

        Returns:
            The current VarContainer with updated data for the selected
            variables.
        """
        for var in self.selected(list(name_data_dict.keys())):
            new_data = name_data_dict.get(var.name)
            var.with_data(new_data, inplace=True)
        return self

    def substitute_metadata(self, **name_metadata_dict) -> Self:
        """Substitute metadata for selected variables.

        Args:
            **name_metadata_dict: A dictionary with variable names as
                keys and new metadata as values.

        Returns:
            The current VarContainer with updated metadata for the
            selected variables.
        """
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
        """Retrieve all variables contained within the VarContainer.

        Returns:
            A tuple of the contained OptimizationVariables.

        Raises:
            TypeError: If the internal variable storage is not a tuple
                or list.
        """
        if isinstance(self._variables, tuple):
            return self._variables
        elif isinstance(self._variables, list):
            return tuple(self._variables)
        else:
            raise TypeError("Something went wrong with the type of the variables.")

    @property
    def constants(self):
        """Retrieve all variables marked as constants from the container.

        Returns:
            A VarContainer containing only the variables marked as
            constants.
        """
        return VarContainer(
            tuple(variable for variable in self.variables if variable.is_constant)
        )

    @property
    def decision_variables(self):
        """Retrieve all variables not marked as constants from the container.

        Returns:
            A VarContainer containing only the variables not marked as
            constants.
        """
        return VarContainer(
            tuple(variable for variable in self.variables if not variable.is_constant)
        )

    @property
    def names(self):
        """Get the names of all variables contained within the VarContainer.

        Returns:
            A tuple containing the names of all variables.
        """
        return tuple(var.name for var in self.variables)

    def __radd__(self, other) -> Self:
        """Support the right-adding of this container to another object.

        Returns:
            The current VarContainer instance.
        """
        return self

    def __add__(self, other) -> Self:
        """Add another container or variable to this container.

        Args:
            other: The VarContainer or OptimizationVariable to be added.

        Returns:
            A new VarContainer instance containing the combined
            variables.

        Raises:
            NotImplementedError: If the addition operation is not
                defined for the `other` type.
        """
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
        """Create an iterator over the contained variables.

        Returns:
            An iterator for the container's variables.
        """
        for variable in self.variables:
            yield variable

    def __len__(self):
        """Get the number of variables contained within the VarContainer.

        Returns:
            The number of contained variables.
        """
        return len(self._variables)

    def __getitem__(self, key) -> Union[OptimizationVariable, Self]:
        """Retrieve a contained variable by index, slice, or name.

        Args:
            key: The index, slice, or name used to retrieve the
                variable(s).

        Returns:
            The requested OptimizationVariable or a new VarContainer
            with the sliced variables.

        Raises:
            KeyError: If a variable with the given name is not found.
            NotImplementedError: If the key type is not supported.
        """
        if isinstance(key, int) and isinstance(self.variables, tuple):
            assert 0 <= key < len(self.variables), f"Index {key} is out of bounds."
            return self.variables[key]
        elif isinstance(key, slice) and isinstance(self.variables, tuple):
            return VarContainer(self.variables[key])
        elif isinstance(key, str) and self.variables_hashmap is not None:
            res = self.variables_hashmap.get(key)
            if res is None:
                raise KeyError(f"OptimizationVariable {key} not found.")
            return res
        else:
            raise NotImplementedError

    def fix(
        self,
        variables_to_fix: list[str],
        hook: Optional[Hook] = None,
    ) -> None:
        """Fix the values of the specified variables and optionally apply a hook.

        Args:
            variables_to_fix: A list of variable names to be fixed.
            hook: An optional Hook object to be applied to the fixed
                variables.
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
        """Unfix the values of the specified variables and optionally apply a hook.

        Args:
            variables_to_unfix: A list of variable names to be unfixed.
            hook: An optional Hook object to be applied to the unfixed
                variables.
        """
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
    """Represent a nested function within the optimization process.

    This class extends the OptimizationVariable to represent a nested function, including variables on which it depends.

    Args:
        nested_variables: A container for variables that the nested
            function depends on. Defaults to an empty VarContainer.
    """

    nested_variables: VarContainer = field(default_factory=lambda: VarContainer([]))

    def substitute_parameters(self, **parameters):
        """Substitute parameters within the nested variables.

        Args:
            **parameters: A dictionary where keys are variable names and
                values are the new data to substitute.
        """
        self.nested_variables.substitute_data(**parameters)
        self.nested_variables.substitute_metadata(**parameters)

    def with_data(self, new_data, inplace=True):
        """Associate new data with the nested function.

        Args:
            new_data: The new data to associate with the nested
                function.
            inplace: If True, modify the nested function in place;
                otherwise, return a new instance. Defaults to True.

        Returns:
            The nested function with the updated data (self or new
            instance).
        """
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
        """Associate new metadata with the nested function.

        Args:
            new_metadata: The new metadata to associate with the nested
                function.
            inplace: If True, modify the nested function in place;
                otherwise, return a new instance. Defaults to True.

        Returns:
            The nested function with the updated metadata (self or new
            instance).
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

    def renamed(self, new_name: str, inplace=True) -> Self:
        """Rename the nested function.

        Args:
            new_name: The new name for the nested function.
            inplace: If True, modify the nested function in place;
                otherwise, return a new instance. Defaults to True.

        Returns:
            The nested function with the updated name (self or new
            instance).
        """
        if inplace:
            self.name = new_name
            return self
        else:
            return NestedFunction(
                name=new_name,
                dims=self.dims,
                data=self.data,
                metadata=self.metadata,
                is_constant=self.is_constant,
                nested_variables=self.nested_variables,
            )

    def as_constant(self, inplace=True):
        """Mark the nested function as a constant.

        Args:
            inplace: If True, modify the nested function in place;
                otherwise, return a new instance marked as constant.
                Defaults to True.

        Returns:
            The nested function marked as a constant (self or new
            instance).
        """
        if self.is_constant:
            return self
        else:
            if inplace:
                self.is_constant = True
                return self
            else:
                return NestedFunction(
                    name=self.name,
                    dims=self.dims,
                    data=self.data,
                    metadata=self.metadata,
                    is_constant=True,
                    nested_variables=self.nested_variables,
                )

    def as_decision_variable(self, inplace=True):
        """Mark the nested function as a decision variable.

        Args:
            inplace: If True, modify the nested function in place;
                otherwise, return a new instance marked as a decision
                variable. Defaults to True.

        Returns:
            The nested function marked as a decision variable (self or
            new instance).
        """
        if not self.is_constant:
            return self
        else:
            if inplace:
                self.is_constant = False
                return self
            return NestedFunction(
                name=self.name,
                dims=self.dims,
                data=self.data,
                metadata=self.metadata,
                is_constant=False,
                nested_variables=self.nested_variables,
            )


@dataclass
class FunctionWithSignature:
    """Wraps a callable function and manage its signature for the optimization process.

    This class parses the function's signature to ensure that it complies with the expected format for the optimization procedure.

    Args:
        func: The callable function to be wrapped.
        variables: A container of variables that the function operates
            on. Defaults to an empty VarContainer.
        is_objective: Flag indicating if the function is an objective
            function. Defaults to False.
        metadata: Additional metadata for the function. Defaults to
            None.
    """

    func: Callable
    variables: VarContainer = field(default_factory=lambda: VarContainer([]))
    is_objective: bool = False
    metadata: Any = None

    def __post_init__(self) -> None:
        """Initialize the function wrapper by parsing its signature.

        Raises:
            ValueError: If there are unknown variables in the signature
                that are not in the variable container.
        """
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
        """Determine if there are constants awaiting substitution.

        Returns:
            True if there are constants to substitute, False otherwise.
        """
        return not (self.constants_to_substitute == [])

    @property
    def constants_to_substitute(self):
        """List the names of constants that need to be substituted.

        Returns:
            A list of constant names that require substitution.
        """
        return [var.name for var in self.variables.constants if var() is None]

    def __call__(
        self, *args, with_metadata: bool = False, raw_eval: bool = False, **kwargs
    ) -> Any:
        """Call the wrapped function with the provided arguments and keyword arguments.

        Args:
            *args: Positional arguments to be passed to the function.
            with_metadata: If True, pass metadata to the function.
                Defaults to False.
            raw_eval: If True, evaluate the function with raw data.
                Defaults to False.
            **kwargs (Dict[str, Any]) : Keyword arguments to be passed to the function.

        Returns:
            The result of the function call.

        Raises:
            ValueError: If not all required constants have been
                substituted or if excess arguments are provided.
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
            else:
                dvar.with_data(args[0])
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
        """Get the signature of the wrapped function.

        Returns:
            A tuple representing the function's signature.
        """
        return tuple(self.__signature)

    @property
    def occupied(self) -> tuple:
        """Get the names of constants in the variables.

        Returns:
            A tuple containing the names of all constant variables.
        """
        return tuple(self.variables.constants.names)

    @property
    def free_placeholders(self) -> tuple:
        """Returns a list of free variables of the current function.

        Free variables are the arguments that
        are not defaulted and do not have a corresponding value.
        This method uses the signature of the function
        and the default parameters keys to determine the free variables.

        Returns:
            tuple: A list of free variables of the current function.
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
        replace: bool = False,
    ) -> Self:
        """Declare variables for the function, optionally replacing existing ones.

        Args:
            variables: The variables to declare for the function.
            replace: If True, replace any existing variables. Defaults
                to False.

        Returns:
            The instance of the function with declared variables.

        Raises:
            AssertionError: If unknown parameters are provided.
        """
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
        """Set parameters for the function.

        Args:
            **kwargs (Dict[str, Any]): Keyword arguments representing the parameters to
                set.

        Raises:
            ValueError: If unknown parameters are encountered.
        """
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
        """Fix variables and optionally set their data and metadata.

        Args:
            variables_to_fix: A list of variable names to be fixed.
            data_dict: Optional dictionary with data to set for fixed
                variables.
            metadata_dict: Optional dictionary with metadata to set for
                fixed variables.
            hook: Optional hook to be applied to fixed variables.
        """
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
        """Unfix variables and optionally apply a hook.

        Args:
            variables_to_unfix: A list of variable names to be unfixed.
            hook: Optional hook to be applied to unfixed variables.
        """
        self.variables.unfix(variables_to_unfix, hook=hook)

    def __parse_signature(self, func: Callable) -> Tuple[str]:
        """Parse the signature of the given function.

        Args:
            func: The function to parse the signature for.

        Returns:
            A tuple representing the parsed signature.

        Raises:
            ValueError: If the function has undefined positional or
                keyword arguments.
        """
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

    def __radd__(self, other: Self) -> FuncContainer:
        """Support the right-adding of this function to another object to form a FuncContainer.

        Args:
            other: The object to add this function to.

        Returns:
            A FuncContainer containing this function.
        """
        return FuncContainer((self,))

    def __add__(self, other: Self) -> Optional[FuncContainer]:
        """Add another function or container to this function to form a FuncContainer.

        Args:
            other: The function or FuncContainer to be added.

        Returns:
            A FuncContainer including this function and `other`, or None
            if `other` is not of a supported type.
        """
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
