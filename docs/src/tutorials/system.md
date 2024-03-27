This tutorial provides a comprehensive guide on leveraging the [`system.py`][regelum.system] module. 
The module offers a framework for defining abstract and concrete classes that represent dynamical systems, 
tailored for reinforcement learning and control engineering tasks.

## Step-by-step overview

Creating a custom system within the Regelum framework involves a few systematic steps:

- Inherit from the abstract base class [`System`][regelum.system.System].
- Declare essential class attributes, such as state space dimension ( `_dim_state` ), input dimension ( `_dim_inputs` ), and observation dimension ( `_dim_observation` ), etc. 
The full list of attributes is provided [below](#utilization-of-private-variables).
- Define the [`_compute_state_dynamics()`][regelum.system.SystemInterface._compute_state_dynamics] method to encapsulate 
the right-hand side of the system's differential equations, enabling the modeling of state transitions in response to control inputs.
- Optionally, override the [`_get_observation()`][regelum.system.SystemInterface._get_observation] method if the observation model differs from the direct state representation, to detail the process of deriving observations from the state.

### Utilization of Private Variables

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
for their utility, particularly in enhancing the interpretability of results, such as labeling axes in plots generated from callbacks TODO ADD LINK TO CALLBACKS.

## Example 1: [Kinematic three-wheeled robot](../systems/3wrobot_kin.md)

The motion of a non-holonomic robot is governed by the following differential equations:


{% include 'systems/3wrobot_kin/state_dynamics.md' %}

where the state and the observation are

{% include 'systems/3wrobot_kin/state_and_observation.md' %}

the action $\action$ is 

{% include 'systems/3wrobot_kin/action.md' %}

The components of the action are subject to the following constraints:

{% include  'systems/3wrobot_kin/action_bounds.md' %}

Below we provide the code that implements the defined system.

```python
from regelum.system import System
import numpy as np

from regelum.utils import rg  # Essential for array computations

class MyThreeWheeledRobotKinematic(System):
    """Kinematic three-wheeled robot system implementation. """

    # These private variables are leveraged 
    # by other components within the codebase.

    # While optional, naming variables 
    # enhance usability, especially for plotting.

    _name = 'ThreeWheeledRobotKinematic'
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
        Dstate = rg.zeros(self._dim_state, prototype=state) #
        
        # Element-wise calculation of the Dstate vector 
        # based on the system's differential equations
        Dstate[0] = inputs[0] * rg.cos(state[2])  # v * cos(vartheta)
        Dstate[1] = inputs[0] * rg.sin(state[2])  # v * sin(vartheta)
        Dstate[2] = inputs[1]                     # omega

        return Dstate
```

## Example 2: [Dynamic three-wheeled robot](../systems/3wrobot_dyn.md)

The motion of a non-holonomic robot is governed by the following differential equations:


{% include 'systems/3wrobot_dyn/state_dynamics.md' %}

where the state and the observation are

{% include 'systems/3wrobot_dyn/state_and_observation.md' %}

the action $\action$ is 

{% include 'systems/3wrobot_dyn/action.md' %}

The components of the action are subject to the following constraints:

{% include  'systems/3wrobot_dyn/action_bounds.md' %}

Below we provide the code that implements the defined system.

``` python
class MyThreeWheeledRobotDynamic(System):
    """ Three-Wheeled Robot with dynamic actuators. """

    _name = "three-wheeled-robot"
    _system_type = "diff_eqn"
    _dim_state = 5
    _dim_inputs = 2
    _dim_observation = 5
    _parameters = {"m": 10, "I": 1}
    _observation_naming = _state_naming = [
        "x_c", 
        "y_c", 
        "angle", 
        "linear_velocity", 
        "angular_velocity".
    ]
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
```

## Working with Composed Systems

A composed system is an assembly of multiple interacting subsystems that form a larger, more complex system. 
It allows for modular design and can encapsulate the functionality of various independent components into one cohesive unit. 
The key advantage of using composed systems is the ability to design complex behavior by manipulating individual, simpler systems. 
The difference between a composed system and a single system lies in the composed system's capability to route outputs of underlying 
systems as inputs to other systems, leading to more flexible and scalable designs. 
 
A composed system consists of multiple subsystems, where the output of one subsystem can become the input to another. 

To create a composed system in Regelum, you need to specify the interaction between subsystems using the `@` operator. 
This operator effectively connects the output of one system to the input of another. 
The specifics of the connection can be controlled by specifying how the routing of inputs and outputs should be performed.

The state of a composed system is often an assembly of the states of the individual subsystems it consists of. 
However, the ordering of these states may not align with the expectations of downstream processes or system requirements. 
To address this, the [`permute_state`][regelum.system.ComposedSystem.permute_state] method can be utilized to rearrange the states into the correct order.

### Example 3: Combining `Integrator` with [kinematic robot][example 1]

Our main goal in this example is to derive a system from [example 2] by composing the system from 
[example 1]
with the simple `Integrator` system that is represented by the following dynamics:

$$
\diff 
\left(
\begin{array}{c}
    v \\  
    \omega
\end{array} 
\right) 
= 
\left(
\begin{array}{c}
    \frac{F}{m} \\
    \frac{M}{I}
\end{array}
\right) \diff t
$$

where state, observation and action are

$$\state = \obs = \left(
\begin{array}{c}
    v \\  
    \omega
\end{array} 
\right), \qquad \action = \left(\begin{array}{c} F \\ M\end{array}\right)$$

```python 
from regelum.system import System
from regelum.__utilities import rg  # Essential for array computations

class Integrator(System):

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
```

When we compose it with the class `MyThreeWheeledRobotKinematic` from [example 1]
using the `@` operator, we get a new composed system . In this case, the state of the composed system 
merges the individual states of the `Integrator` and [example 1], potentially requiring permutation to align with the dynamics 
of the intended `MyThreeWheeledRobotDynamic` system from [example 2].

``` python 
composed_system = Integrator() @ MyThreeWheeledRobotKinematic()
```

The resulting state of the composed system is:

\begin{equation}\label{eq:state_composed}
    \state_{\text{composed}} = \left(
\begin{array}{c}
    v \\
    \omega \\
    x \\
    y \\
    \vartheta
\end{array}
\right)
\end{equation}

This occurs because the `@` operator concatenates the state of the right system to the state of the left system.

However, for the state to represent the `MyThreeWheeledRobot` system correctly, as specified in [example 2], it must be formatted as

\begin{equation}{\label{eq:state_composed_perm}}
\left(
\begin{array}{c}
    x \\
    y \\
    \vartheta \\
    v \\
    \omega 
\end{array}
\right)
\end{equation}

To achieve the correct state format, we utilize the [`permute_state`][regelum.system.ComposedSystem.permute_state] method to rearrange the states:

``` python
# Create the composed system
composed_system = Integrator() @ MyThreeWheeledRobotKinematic()
# Permute the states to the correct order
composed_system = composed_system.permute_state([3, 4, 0, 1, 2])
```

To remember the operation of `permute_state`, consider this simple mnemonic rule:

"Each number moves to the house with its own number above the door."

Here's how you apply the rule for the list `[3, 4, 0, 1, 2]`:

1. The number at position 0 (first) moves to house number 3.
2. The number at position 1 (second) moves to house number 4.
3. The number at position 2 (third) moves to house number 0.
4. The number at position 3 (fourth) moves to house number 1.
5. The number at position 4 (fifth) moves to house number 2.

Following this procedure, the original state representation $\eqref{eq:state_composed}$ is permuted to 
$\eqref{eq:state_composed_perm}$
using the permutation encoded by `[3, 4, 0, 1, 2]`.

### Example 4: Composing with ConstantReference for Targeted Stabilization

In scenarios where we aim to stabilize the system at a particular state instead of the origin, 
composing the system with ConstantReference provides a way to achieve that. 
For instance, to park a robot at a designated point (e.g., (1, 1, 0)), we can construct a system whose observation is the difference between the state and a constant reference.

```python 
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
```

Let us create the composed system as follows: 


``` python
# Compose the robot system with the constant reference
composed_system = ComposedSystem(
    sys_left=ThreeWheeledRobotKinematic(),
    sys_right=ConstantReference(),
    io_mapping=None,
    output_mode="right",
    inputs_naming=system.inputs_naming,
    state_naming=system.state_naming,
    observation_naming=[s + "-ref" for s in system.state_naming],
    action_bounds=system.action_bounds,
)
```
The system has the same state as `MyThreeWheeledRobotKinematic()` from [example 1 test]: 

$$
\state = \left(
    \begin{array}{c}
        x \\
        y \\
        \vartheta
    \end{array}
\right)
$$

and observation 

$$
\obs = \left(
    \begin{array}{c}
        x - 1 \\
        y - 1\\
        \vartheta
    \end{array}
\right)
$$

  [example 1]: #example-1-kinematic-three-wheeled-robot
  [example 2]: #exmaple-2-dynamic-three-wheeled-robot