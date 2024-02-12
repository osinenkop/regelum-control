## Regelum Framework Tutorial: Implementing and Simulating a Kinematic Point System

TODO: OBSERVATION AND STABILIZE to (1, 1) AND ADD NOISE TO OBSERVATION AND STABILZIE IT WITH KALMAN FILTER VIA NEW OBSERVER

This tutorial will walk you through the process of implementing a simple kinematic point system within the Regelum framework, defining a basic stabilizing control action, instantiating a CasADi simulator, running a simulation loop, and finally plotting the trajectory of the system using matplotlib.

### Kinematic Point System

A kinematic point is a basic model that represents the motion of a point mass in a 2D space influenced by velocity inputs. In this system, the state typically comprises the point's position coordinates, while the inputs control the point's velocity along respective axes.

Here's a brief description of the attributes within the Kinematic Point System implemented in Regelum:

- `_name`: Identifier name for the system.
- `_system_type`: Marks the system as a differential equation (`"diff_eqn"`).
- `_dim_state`: Dimension of the state, `2` for 2D space (x, y coordinates).
- `_dim_inputs`: Dimension of the inputs, `2` for velocity controls along each axis (v_x, v_y).
- `_dim_observation`: Dimension of the observation, which is the same as the state dimension in this simple case.
- `_observation_naming` & `_state_naming`: Names for state dimensions for better interpretability.
- `_inputs_naming`: Names for input dimensions.
- `_action_bounds`: Defines the bounds for each action (velocity input) dimension to enforce feasible velocity ranges.

### Implementing Kinematic Point System

TODO: ADD SYSTEM DESCRIPTION

Here's an example of how a Kinematic Point system can be defined:
``` python
from regelum.system import System
import numpy as np

class KinematicPoint(System):
    _name = "kinematic-point"
    _system_type = "diff_eqn"
    _dim_state = 2
    _dim_inputs = 2
    _dim_observation = 2
    _observation_naming = _state_naming = ["x", "y"]
    _inputs_naming = ["v_x", "v_y"]
    _action_bounds = [[-10.0, 10.0], [-10.0, 10.0]]

    def _compute_state_dynamics(self, time, state, inputs):
        return inputs  # The velocity inputs directly define the rate of change of position.
```

### Stabilizing Action Function

TODO: DEMONSTRATE THAT -x IS THE SOLUTION

To achieve stabilization of the kinematic point, we'll define a function `get_action()` that produces control inputs designed to drive the system's state (position) towards the origin (i.e., achieving stabilization). This is a simple negative feedback control law.

``` python
def get_action(state):
    return -state  # Stabilizing action: control input is the negative of the system state.
```

### Instantiating CasADi Simulator

Next, we'll instantiate a CasADi simulator, providing the kinematic point system and the initial conditions.

``` python
from regelum.simulator import CasADi

# Define the initial state (initial position of the kinematic point).
initial_state = np.array([[2.0], [2.0]])  # Start at position (2, 2)

# Initialize the kinematic point system.
kinematic_point = KinematicPoint()

# Instantiate a simulator for the kinematic point system.
simulator = CasADi(system=kinematic_point, state_init=initial_state, time_final=10, max_step=0.1)
```

### Running the Simulation Loop

TODO: get_sim_step_data

With the simulator ready, we can now run the simulation loop, applying the stabilizing action at each step and recording the system state history.
``` python

state_history = [initial_state.flatten()]  # Store the initial state.

for _ in range(int(simulator.time_final / simulator.max_step)):
    action = get_action(simulator.state)  # Compute the action based on the current state.
    simulator.receive_action(action)  # Provide the action to the simulator.
    simulator.step()  # Perform one simulation step.
    state_history.append(simulator.state.flatten())  # Store the state after the step.

state_history = np.array(state_history)  # Convert history to numpy array for plotting.
```

### Plotting the Trajectory

TODO: PLOT x(t), y(t)

Finally, we can visualize the trajectory of the kinematic point with matplotlib.

``` python
import matplotlib.pyplot as plt

# Plot the trajectory of the kinematic point over time.
plt.figure(figsize=(8, 8))
plt.plot(state_history[:, 0], state_history[:, 1], marker='o')
plt.title('Trajectory of Kinematic Point')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.axis('equal')
plt.show()
```