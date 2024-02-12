.. _rc_singleton_tutorial:

**********************************
The **rg** Singleton Tutorial
**********************************

What is the **rg** Singleton?
===========================

In computational frameworks involving matrix algebra, developers often switch between different computation backends such as NumPy, PyTorch, 
and CasADi based on the problem needs. NumPy is suitable for CPU computations, PyTorch is preferred for tensor operations with GPU support, 
and CasADi is used for symbolic computation and automatic differentiation. Manually adapting the code to support all these backends can be cumbersome and error-prone.
This is where the **rg** singleton comes into play, providing a unified interface to perform matrix operations across all these types comfortably.

The Need for Abstraction
========================

Consider this code snippet where the function `some_computation` is designed to work with NumPy arrays:

.. code-block:: python

    import torch as th
    def some_computation(x, y):
        return th.dot(x, y)  # dot product using PyTorch

The function operates as expected with Torch-type arguments:

.. code-block:: python

    A = th.tensor([1, 2])
    B = th.tensor([5, 6])
    some_computation(A, B)
    # output: tensor(17)

However, if we use CasADi-type arguments:

.. code-block:: python

    import casadi
    A_casadi = casadi.DM([1, 2])
    B_casadi = casadi.DM([5, 6])
    some_computation(A_casadi, B_casadi)
    # This will raise a TypeError: argument 'input' (position 1) must be Tensor, not DM

To resolve this issue, a tool that generalizes matrix operations across argument types is highly beneficial. With such a tool, 
the same function could effortlessly switch between different argument types, enhancing code modulation and simplicity.

Introducing **rg** in Action
==========================

With **rg**, the initial code snippet can be rewritten to handle any type of argument seamlessly:

.. code-block:: python

    from regelum.__utilities import rg
    def some_computation(x, y):
        return rg.dot(x, y)  # dot product generalized by **rg**

Using `rg.dot`, operations can be executed as expected without type errors:

.. code-block:: python

    some_computation(A_casadi, B_casadi)
    # output: DM(17)

Type System and Automatic Inference
===================================

The **rg** singleton uses a type inference system that automatically deduces the type based on provided arguments. NumPy uses CPU arrays, 
PyTorch leverages GPU-accelerated tensors, and CasADi works with symbolic expressions. 
When multiple types are involved in a computation, **rg** will choose the type for the operation, ensuring consistency across calculations.

Meta-Programming in RCTypeHandler
=================================

**rg** instantiates from the `RCTypeHandler` class, employing a meta-programming technique. Upon instantiation, `RCTypeHandler` enables automatic type inference 
by decorating each method of **rg** with functions that perform type-checking and substitute and appropriate `rc_type` as an argument to each method call, 
thus managing the method's behavior according to argument types without the need for explicit type checks.

Using **rg** for Array Creation
=============================

Several methods within the **rg**` singleton are crucial for creating new arrays, such as `zeros`, `ones`, and `array`. Here's how the `prototype` keyword argument is pivotal:

.. code-block:: python

    R = rg.zeros((3, 3), prototype=some_np_array)  # NumPy zeros matrix of shape (3, 3)
    G = rg.ones((2, 2), prototype=some_torch_tensor)  # Torch ones matrix of shape (2, 2)
    L = rg.zeros((2, 3), prototype=some_casadi_MX_matrix)  # CasADi symbolic zeros matrix of shape (2, 3)
    M = rg.array([1,2,3], prototype=some_casadi_DM_matrix)  # CasADi numeric (DM) matrix of shape casadi.DM([1,2,3])

The `prototype` argument enables **rg** to infer the proper type for the array creation without explicit user specification.

Expanding **rg** with Custom Functions
====================================

To extend the capabilities of **rg**, users can integrate custom functions. It is essential to include `rc_type` in the function signature for proper integration:

.. code-block:: python

    def my_custom_function(arg1, arg2, rc_type=rg.NUMPY):
        # Function implementation that leverages 'rc_type' for type-specific operations
        pass

Demonstration of **rg** in System Dynamics
========================================

Let's consider the dynamics of a three-wheeled robot, where the **rg** singleton enables computations with symbolic,

.. code-block:: python

    from regelum.system import System
    import numpy as np

    from regelum.__utilities import rg  # Essential for array computations

    # Implementation of the three-wheeled robot system with **rg**
    class ThreeWheeledRobotKinematic(System):
        ...  # The class definition

        def _compute_state_dynamics(self, time, state, inputs):
            # Calculate the robot's state dynamics using **rg**
            Dstate = rg.zeros(self._dim_state, prototype=state)
            Dstate[0] = inputs[0] * rg.cos(state[2])
            Dstate[1] = inputs[0] * rg.sin(state[2])
            Dstate[2] = inputs[1]
            
            return Dstate

The **rg** singleton shines in its ability to handle the `_compute_state_dynamics` function with argument 
types that can vary dramatically during runtime. For instance, 
the `CasADi` ODE solver demands a symbolic prototype to function correctly, 
whereas reinforcement learning algorithms might need to propagate a torch tensor through the same function.

The rg singleton facilitates this by allowing the `_compute_state_dynamics` function to operate seamlessly 
with these different types of arguments. Whether we need to work with symbolic representations 
for simulation or actual data types like torch tensors for model-based reinforcement learning methods, 
the rg singleton ensures that the underlying computations remain consistent and type-correct. 
This flexibility is crucial for successfully integrating diverse computational requirements, 
such as in `Actor-Critic` (`RPO`) algorithms, where the same function is used across different contexts and with various data types.

It's important to note, however, that if your work doesn't require symbolic computations or optimizations, 
you have the option to stick with pure numpy or torch functions. 
This is particularly relevant when you're leveraging numpy/scipy-exclusive packages for your optimization and simulation tasks.