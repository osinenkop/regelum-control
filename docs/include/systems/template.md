


{% set description_path = 'systems/' ~ system ~ '/description.md' %}
{% set state_dynamics_path = 'systems/' ~ system ~ '/state_dynamics.md' %}
{% set state_and_observation_path = 'systems/' ~ system ~ '/state_and_observation.md' %}
{% set action_path = 'systems/' ~ system ~ '/action.md' %}
{% set action_bounds_path = 'systems/' ~ system ~ '/action_bounds.md' %}


## Description

{% include description_path %}


## System dynamics

The dynamics are captured by the following differential equations, with the physical meaning of each entity described in the subsequent sections:

{% include 'preamble.md' %}

{% include state_dynamics_path %}

{% if is_include_parameters %}
## System parameters
{% set parameters_path = 'systems/' ~ system ~ '/parameters.md' %}
{% include parameters_path %}
{% endif %}

## State and observation

The state $\state$ of the system and the observation $\obs$ are

{% include state_and_observation_path %}

???+ note

    In reinforcement learning and control systems, observations are the perceivable parts of the system's state and control 
    inputs that are accessible at a particular point in time. 
    These observations are used by an agent or a controller to make decisions or by evaluation tools to assess the system's performance.

## Action

The action $\action$ (e.g. the control inputs) is

{% include action_path %}

## Action bounds

The components of the action are subject to the following constraints:

{% include action_bounds_path %}

## Reference API


???+ note

    The class [`regelum.system.{{ api_reference }}`][regelum.system.{{ api_reference }}] defines the system dynamics but does not perform any simulation mechanics itself. 
    It contains only the update rule as defined in [system dynamics](#system-dynamics) section. The actual simulation routine is executed by a 
    separate [`Simulator`][regelum.simulator.Simulator] class, which leverages the update rule to create 
    the corresponding integration scheme. Commonly, we use a simulator [`CasADi`][regelum.simulator.CasADi] that formulates a 
    Runge-Kutta integration scheme on the flow. An example of employing the simulator is provided [below](#usage-example).


::: regelum.system.{{ api_reference }}
    options:
      heading_level: 3
      show_root_full_path: true
      show_root_members_full_path: true

### Usage example

    
???+ example

    To properly initialize a [`regelum.system.{{ api_reference }}`][regelum.system.{{ api_reference }}] system along with the [`CasADi`][regelum.simulator.CasADi] 
    simulator class, which integrates a Runge-Kutta scheme, from the initial state coordinates of $({{ state_init }})$, integration timestep of ${{pred_step_size}}$ seconds, 
    and a final time of simulation ${{ time_final }}$ seconds, we can use the following snippet:

    ```python
    import numpy as np
    from regelum.system import {{ api_reference }}
    from regelum.simulator import CasADi

    system = {{ api_reference }}()
    simulator = CasADi(
        system=system,
        max_step={{ pred_step_size }},
        state_init=np.array([{{ state_init_py }}]),
        time_final={{ time_final }},
    )
    ```