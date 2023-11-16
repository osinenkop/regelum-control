**************************************************
Tutorial: Implementing Custom Systems in Regelum
**************************************************

This tutorial provides a comprehensive guide on leveraging the `system.py` module from the Regelum package. The module offers a framework for defining abstract and concrete classes that represent dynamical systems, tailored for reinforcement learning and control engineering tasks.

Implementing Your Own System
============================

Creating a custom system within the Regelum framework involves a few systematic steps:

#. Inherit from the abstract base class `System`.
#. Declare essential class attributes, such as state space dimension (`_dim_state`), input dimension (`_dim_inputs`), and observation dimension (`_dim_observation`).
#. Define the `_compute_state_dynamics()` method to encapsulate the right-hand side of the system's differential equations, enabling the modeling of state transitions in response to control inputs.
#. Optionally, override the `_get_observation()` method if the observation model differs from the direct state representation, to detail the process of deriving observations from the state.

Utilization of Private Variables
--------------------------------

The following private variables are crucial for the internal mechanics of other code components within Regelum:

- `_name`
- `_system_type`
- `_dim_state`
- `_dim_inputs`
- `_dim_observation`
- `_observation_naming`
- `_state_naming`
- `_inputs_naming`
- `_action_bounds`
- `_parameters` 

Naming variables such as `_observation_naming`, `_state_naming`, and `_inputs_naming`, while optional, are highly recommended 
for their utility, particularly in enhancing the interpretability of results, such as labeling axes in plots generated from callbacks.

.. _example 1:

Example 1: Non-holonomic Robot
------------------------------

The motion of a non-holonomic robot is governed by the following differential equations:

.. math::

    \begin{aligned}
        &\dot{x}_{с} = v \cos(\vartheta), \\
        &\dot{y}_{c} = v \sin(\vartheta), \\
        &\dot{\vartheta} = \omega,
    \end{aligned}

where:
  - :math:`{x}_{c}` is the rate of change of the robot's x-position.
  - :math:`y_{c}` is the rate of change of the robot's y-position.
  - :math:`\vartheta` is the robot's orientation.
  - :math:`v` is the linear velocity input.
  - :math:`\omega` is the angular velocity input.

The corresponding Python implementation uses the robot's state directly for the observation, thus no `_get_observation()` method override is necessary. The state and observation are three-dimensional:

.. math::

    (x_{c}, y_{c}, \vartheta)

These components represent the robot's state in the plane.

Control inputs :math:`(v, \omega)` are utilized in the code as follows:

.. code-block:: python

    from regelum.system import System
    import numpy as np

    from regelum.__utilities import rg  # Essential for array computations

    class ThreeWheeledRobotNI(System):
        """ Non-holonomic three-wheeled robot system implementation. """

        # These private variables are leveraged by other components within the codebase.
        # While optional, naming variables enhance usability, especially for plotting.

        _name = 'ThreeWheeledRobotNI'
        _system_type = 'diff_eqn'
        _dim_state = 3
        _dim_inputs = 2
        _dim_observation = 3
        _observation_naming = _state_naming = ["x_rob", "y_rob", "vartheta"]
        _inputs_naming = ["v", "omega"]
        _action_bounds = [[-25.0, 25.0], [-5.0, 5.0]] 

        def _compute_state_dynamics(self, time, state, inputs):
            """ Calculate the robot's state dynamics. """

            # Placeholder for the right-hand side of the differential equations
            Dstate = rg.zeros(self._dim_state, prototype=state)
            
            # Element-wise calculation of the Dstate vector based on the system's differential equations
            Dstate[0] = inputs[0] * rg.cos(state[2])  # v * cos(vartheta)
            Dstate[1] = inputs[0] * rg.sin(state[2])  # v * sin(vartheta)
            Dstate[2] = inputs[1]                     # omega

            return Dstate

.. _example 2:

Example 2: Three-Wheeled Robot with Dynamical Actuators
--------------------------------------------------------

The `ThreeWheeledRobot` class embodies a three-wheeled robot system equipped with dynamic actuators. 

.. math::
    \begin{array}{ll}
        \dot{x}_c      & = v \cos(\theta), \\
        \dot{y}_c      & = v \sin(\theta), \\
        \dot{\theta}   & = \omega, \\
        \dot{v}        & = \left( \frac{1}{m} F \right), \\
        \dot{\omega}   & = \left( \frac{1}{I} M  \right)
    \end{array}

Variable Definitions:

- :math:`x_c`: State-coordinate [m]
- :math:`y_c`: Observation-coordinate [m]
- :math:`\theta`: Turning angle [rad]
- :math:`v`: Linear velocity [m/s]
- :math:`\omega`: Angular velocity [rad/s]
- :math:`F`: Pushing force [N]
- :math:`M`: Steering torque [Nm]
- :math:`m`: Robot mass [kg]
- :math:`I`: Robot moment of inertia around the vertical axis [kg m\ :sup:`2`]

The system state and control inputs are described by:

- :math:`state = (x_c, y_c, \theta, v, \omega)`
- :math:`inputs = (F, M)`

Parameters such as mass (:math:`m`) and moment of inertia (:math:`I`) are stored within the `_parameters` variable:

.. code-block:: python

    class ThreeWheeledRobot(System):
        """ Three-Wheeled Robot with dynamic actuators. """

        _name = "three-wheeled-robot"
        _system_type = "diff_eqn"
        _dim_state = 5
        _dim_inputs = 2
        _dim_observation = 5
        _parameters = {"m": 10, "I": 1}
        _observation_naming = _state_naming = ["x_c", "y_c", "angle", "linear_velocity", "angular_velocity"]
        _inputs_naming = ["Force", "Momentum"]
        _action_bounds = [[-25.0, 25.0], [-5.0, 5.0]]

        def _compute_state_dynamics(self, time, state, inputs):
            """ Compute the system's state dynamics. """

            # Initialize the right-hand-side of the differential equations
            Dstate = rg.zeros(self._dim_state, prototype=(state, inputs))

            m, I = self._parameters["m"], self._parameters["I"]

            # Compute the state derivatives
            Dstate[0] = state[3] * rg.cos(state[2])  # v * cos(theta)
            Dstate[1] = state[3] * rg.sin(state[2])  # v * sin(theta)
            Dstate[2] = state[4]                     # omega
            Dstate[3] = (1 / m) * inputs[0]          # F / m
            Dstate[4] = (1 / I) * inputs[1]          # M / I

            return Dstate



Working with Composed Systems
=============================

What is a Composed System?
--------------------------

A composed system is an assembly of multiple interacting subsystems that form a larger, more complex system. 
It allows for modular design and can encapsulate the functionality of various independent components into one cohesive unit. 
The key advantage of using composed systems is the ability to design complex behavior by manipulating individual, simpler systems. 
The difference between a composed system and a single system lies in the composed system's capability to route outputs of underlying 
systems as inputs to other systems, leading to more flexible and scalable designs. 
 
A composed system consists of multiple subsystems, where the output of one subsystem can become the input to another. 

Creating a Composed System
--------------------------

To create a composed system in Regelum, you need to specify the interaction between subsystems using the `@` operator. 
This operator effectively connects the output of one system to the input of another. 
The specifics of the connection can be controlled by specifying how the routing of inputs and outputs should be performed.

Examples of Composed Systems
----------------------------

The state of a composed system is often an assembly of the states of the individual subsystems it consists of. 
However, the ordering of these states may not align with the expectations of downstream processes or system requirements. 
To address this, the `permute_state` method can be utilized to rearrange the states into the correct order.

Example 3: Combining `Integrator` with `ThreeWheeledRobotNI`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our main goal in this example is to derive a system from :ref:`Example 2 <example 2>` by composing the system from :ref:`Example 1 <example 1>` 
with the simple `Integrator`` system that is represented by the following dynamics:

.. math::

    \begin{array}{ll}
        \dot{v}        & = \left( \frac{1}{m} F \right), \\
        \dot{\omega}   & = \left( \frac{1}{I} M  \right)
    \end{array}



.. code-block:: python

    from regelum.system import System
    from regelum.__utilities import rg  # Essential for array computations

    class Integrator(System):
        """System yielding Non-holonomic double integrator when composed with kinematic three-wheeled robot."""

        _name = "integral-parts"
        _system_type = "diff_eqn"
        _dim_state = 2
        _dim_inputs = 2
        _dim_observation = 2
        _parameters = {"m": 10, "I": 1}

        def _compute_state_dynamics(self, time, state, inputs):
            Dstate = rg.zeros(
                self.dim_state,
                prototype=(state, inputs),
            )

            m, I = self.parameters["m"], self.parameters["I"]

            Dstate[0] = 1 / m * inputs[0]
            Dstate[1] = 1 / I * inputs[1]

            return Dstate

When we compose it with the `ThreeWheeledRobotNI` from :ref:`example 1 <example 1>`
using the `@` operator, we get a new composed system . In this case, the state of the composed system 
merges the individual states of the `Integrator` and :ref:`ThreeWheeledRobotNI <example 1>`, potentially requiring permutation to align with the dynamics 
of the intended :ref:`ThreeWheeledRobot <example 2>` system.

.. code-block:: python

    # Create the composed system
    composed_system = Integrator() @ ThreeWheeledRobotNI()

The resulting state of the composed system is:

.. math:: 
    (v, \omega, x_c, y_c, \theta)

This occurs because the `@` operator concatenates the state of the right system to the state of the left system.

However, for the state to represent the `ThreeWheeledRobot` system correctly, as specified in :ref:`Example 2 <example 2>`, it must be formatted as:

.. math:: 
    (x_c, y_c, \theta, v, \omega)

To achieve the correct state format, we utilize the `permute_state` method to rearrange the states:

.. code-block:: python

    # Create the composed system
    composed_system = Integrator() @ ThreeWheeledRobotNI()
    # Permute the states to the correct order
    composed_system = composed_system.permute_state([3, 4, 0, 1, 2])

To remember the operation of `permute_state`, consider this simple mnemonic rule:

"Each number moves to the house with its own number above the door."

Here's how you apply the rule for the list `[3, 4, 0, 1, 2]`:

1. The number at position 0 (first) moves to house number 3.
2. The number at position 1 (second) moves to house number 4.
3. The number at position 2 (third) moves to house number 0.
4. The number at position 3 (fourth) moves to house number 1.
5. The number at position 4 (fifth) moves to house number 2.

Following this procedure, the original state representation :math:`(v, \omega, x_c, y_c, \theta)` is permuted to :math:`(x_c, y_c, \theta, v, \omega)` 
using the permutation encoded by `[3, 4, 0, 1, 2]`.


Example 4: Composing with ConstantReference for Targeted Stabilization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In scenarios where we aim to stabilize the system at a particular state instead of the origin, 
composing the system with ConstantReference provides a way to achieve that. 
For instance, to park a robot at a designated point (e.g., (1, 1, 0)), we can construct a system whose observation is the difference between the state and a constant reference.

.. code-block:: python

    # Define the reference state for the robot to stabilize to
    class ConstantReference(System):
    """Subtracts reference from system."""

    name = "constant_reference"
    _system_type = "diff_eqn"
    _dim_state = 0
    _dim_inputs = 3
    _dim_observation = 3
    _parameters = {"reference": np.array([[1.0], [1.0], [0.0]])}

    def _get_observation(self, time, state, inputs):
        return inputs - rg.array(
            self.parameters["reference"], prototype=inputs, _force_numeric=True
        )

    def _compute_state_dynamics(self, time, state, inputs):
        return inputs

Let us create the composed system as follows: 

.. code-block:: python 

    # Compose the robot system with the constant reference
    composed_system = ComposedSystem(
        sys_left=ThreeWheeledRobotNI(),
        sys_right=ConstantReference(),
        io_mapping=None,
        output_mode="right",
        inputs_naming=system.inputs_naming,
        state_naming=system.state_naming,
        observation_naming=[s + "-ref" for s in system.state_naming],
        action_bounds=system.action_bounds,
    )

The system has the same state as `ThreeWheeledRobotNI()` and observation that equals state mimus reference (1, 1, 0).

References
==========

.. [1] W. Abbasi, F. urRehman, I. Shah, "Backstepping based nonlinear adaptive control for the extended nonholonomic double integrator", Kybernetika 53.4 (2017), 578–594.




.. References
.. ==========

.. .. [1] W. Abbasi, F. urRehman, I. Shah, "Backstepping based nonlinear adaptive control for the extended nonholonomic double integrator", Kybernetika 53.4 (2017), 578–594.
