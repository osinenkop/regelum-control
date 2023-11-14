.. _optimizable_package_tutorial:

**********************************
Optimizable Subpackage Tutorial
**********************************

Overview
========

The `optimizable` subpackage is designed to provide a uniform interface for defining and solving both constrained and unconstrained optimization problems within various contexts and systems. This subpackage allows users to seamlessly transition between different optimization engines such as SciPy, Torch, and CasADi based on the problem requirements.

Main Entities
=============

The `optimizable` subpackage consists of several key entities:

- **OptimizationVariable**: Represents a variable within the optimization process that can be either a constant or a decision variable.
- **VarContainer**: A container that manages multiple `OptimizationVariable` instances.
- **NestedFunction**: A variable that represents a nested function and includes information about dependent variables.
- **FunctionWithSignature**: A wrapper for functions within optimization that enforces a specific signature suited for optimization.
- **FuncContainer**: Manages multiple instances of `FunctionWithSignature`.
- **Hook**: A modifier that can be applied to an `OptimizationVariable`.
- **ChainedHook**: A sequence of `Hook` instances applied in a specific order to an `OptimizationVariable`.

Optimizable Class
=================

The centerpiece of the `optimizable` subpackage is the `Optimizable` class. It abstracts the definition of an optimization problem, allowing the use of different optimization engines. The actual optimization engine is chosen automatically based on the 'kind' field of the provided configuration. This design facilitates the handling of both constrained and unconstrained problems across different engines and contexts.

The `Optimizable` class includes several key methods used in setting up the optimization problem:

- **create_variable**: Defines a new optimization variable or a nested function based on provided dimensions, whether it is constant and other properties.
- **register_objective**: Registers an objective function with the optimization which the solving routine will attempt to minimize or maximize.
- **register_constraints**: Adds constraints to the optimization problem that must be satisfied by the solution.
- **register_bounds**: Sets the bounds for decision variables to limit their ranges in the optimization problem.

These methods allow users to construct an optimization problem incrementally before solving it using the chosen optimizer.

Why Do We Need It?
==================

The need for the `optimizable` subpackage arises from common scenarios where varying optimization problems occur that may require different approaches. For example, some problems might be better solved with gradient-based optimization engines like Torch, while others might require constraint programming approaches as provided by CasADi. By having a unified interface, developers can focus on defining their optimization problems without worrying about the specificities of the underlying optimization engine. This approach promotes code reusability and simplifies the transition between different optimization strategies or problems.

Method Descriptions
===================

**create_variable(*dims, name, is_constant=False, like=None, is_nested_function=False, nested_variables=None)**
    This method creates a new optimization variable within the `Optimizable` object. The variable can represent either a standard decision variable or a nested function, which includes other variables it depends on.

**register_objective(func, variables)**
    Registers a function as an objective for optimization. It accepts a callable `func` and a list or `VarContainer` of `OptimizationVariable` objects that the function depends on.

**register_constraints(func, variables, name=None)**
    Adds a new constraint to the optimization problem. Similar to `register_objective`, it wraps the function `func` that defines the constraint and associates the relevant variables.

**register_bounds(variable_to_bound, bounds)**
    This method specifies the lower and upper bounds for a decision variable. It is essential for problems that contain variables with range limitations.

Using the Optimizable Class
===========================

Below are some examples illustrating how to use the `Optimizable` class and its methods to define optimization variables, objectives, constraints, and bounds.

Defining Optimization Variables
-------------------------------

.. code-block:: python

    from your_package.optimizable import Optimizable

    class MyOptimizable(Optimizable):
        def __init__(self, optimizer_config):
            super().__init__(optimizer_config=optimizer_config)

            # Create an optimization variable as a decision variable.
            self.x_var = self.create_variable(1, name='x')

            # Create an optimization variable as a nested function.
            self.nested_var = self.create_variable(
                1,
                name='nested_function',
                is_nested_function=True,
                nested_variables=[self.x_var]
            )

Registering an Objective
------------------------

.. code-block:: python

    # Define an example objective function.
    def my_objective(x):
        return x ** 2 + 3 * x + 2

    # Register objective in MyOptimizable class.
    self.register_objective(
        func=my_objective,
        variables=[self.x_var]
    )

Registering Constraints
-----------------------

.. code-block:: python

    # Define an example constraint function.
    def my_constraint(x):
        return x + 1

    # Register constraint in MyOptimizable class.
    self.register_constraint(
        func=my_constraint,
        variables=[self.x_var],
        name='my_constraint'
    )

Registering Bounds
------------------

.. code-block:: python

    # Define bounds for the x_var variable.
    bounds = np.array([[0, 1]])  # Lower bound 0, upper bound 1

    # Register bounds in MyOptimizable class.
    self.register_bounds(variable_to_bound=self.x_var, bounds=bounds)

Functionality of 'rc' Singleton
==============================

The `rc` singleton provided in the `__utilities.py` module is designed to abstract away differences between numerical and symbolic computation, allowing developers to write type-agnostic code. This functionality enables seamless transition between numeric arrays (NumPy), GPU-accelerated tensors (PyTorch), and symbolic primitives (CasADi). Below is a brief overview of how `rc` can be used in various contexts:

Type-Agnostic Mathematical Functions
------------------------------------

The `rc` singleton provides common mathematical operations that detect and handle the type of their inputs automatically. For instance, to calculate the cosine of an array, tensor, or symbolic variable, one would simply call:

.. code-block:: python

    cos_value = rc.cos(input_variable)

Whether `input_variable` is a NumPy array, Torch tensor, or CasADi symbolic, `rc.cos()` will return the cosine of the input in the corresponding type.

These features of the `rc` singleton simplify the development process by minimizing the need to write type-specific code, enhancing the reusability and generality of the optimization framework.
