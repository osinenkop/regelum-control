# Regelum Framework Overview

Regelum is a Python framework tailored for Reinforcement Learning (RL) and Optimal Control designed to provide an interface for simulating and stabilizing dynamical systems. Below is an overview of the basic concepts within the framework.

## RegelumBase

At the core of Regelum is `RegelumBase`, which acts as an abstraction layer over all objects within the framework. `RegelumBase` equips classes with an interface for callbacks, which are entities that, although residing outside the main execution loop, have access to it. Callbacks facilitate tasks such as logging, graphical saving, and optimization. This is accomplished via the `@apply_callbacks` decorator. 

!!! tip annotate "Related references"

    * [Callbacks](../notebooks/callbacks.md)

## System

Regelum is primarily designed for stabilizing dynamical systems. The key functionality within Regelum is encapsulated in the `System` class, interpretable as a fully static class. It provides the right-hand side (RHS) of the dynamical system equations you wish to implement. Systems can also be composed to create complex dynamics; this composition functionality is a substantial part of Regelum's flexibility. Moreover, Regelum uses `rg` as a type handler, allowing RHS to be computed across all supported backends such as Numpy, Torch, and CasADi. 

!!! tip annotate "Related references"

    * [Implementing your own dynamical system](system.md)
    * [List of available systems](../systems/kin_point.md)

## Simulator

A `Simulator` in Regelum integrates the RHS of a system, providing iterative steps for the simulation. CasADi is the primary backend for simulation. The simulator is an integral component that takes the model of the system's dynamics and approximates its behavior over time. 

!!! tip annotate "Related references"

    * [Run your system in simulator](../notebooks/simulator.md)


## Optimizable

Optimization is a critical component of RL and optimal control problems. Regelum introduces an innovative interface termed `Optimizable`, which allows users to define their optimization problems via a dedicated API. Users can then apply various configurations to the problem such as `tensor` for Torch optimization, `symbolic` for CasADi backend, and `numeric` for NumPy backend.

!!! tip annotate "Related references"

    * [Optimizable and models](../notebooks/optimizable.md)

## Policy

A `Policy` (often an `Optimizable`) within Regelum represents the decision-making process and handles optimization procedures, serving as an "Actor" in RL contexts. The `Policy` interacts with `Model`, the primary backend for decision-making, which in turn dictates the optimization backend to be utilized. Notably, the choice of optimization backend is independent of the simulator's backend, allowing usage of Torch for optimization, even if the simulator relies on a different backend.

!!! tip annotate "Related references"

    * [Run the scenario with a custom policy](../notebooks/policy.md)
    * [Optimizable and models](../notebooks/optimizable.md)
    * [Model Predictive Control](../notebooks/mpc.md)
    * [Policy gradient](../notebooks/policy_gradient.md)

## Critic

The `Critic` is a concept exclusive to RL within Regelum, signifying an entity that provides an estimate of the value of the states or state-action pairs. The critic's role is to assist the policy in enhancing decision-making by evaluating the quality of actions taken, typically updated during the optimization process. [Link to detailed critic explanation]

!!! tip annotate "Related references"

    * [Policy gradient](../notebooks/policy_gradient.md)

## Scenario

The `Scenario` is the orchestrating entity that gathers all other entities like system, simulator, policy, and critic into a cohesive system and initiates the main execution loop. A scenario manages the intricacies of the simulation, optimization, and stabilization procedures, ensuring these components function harmoniously during an RL or optimal control task. 


!!! tip annotate "Related references"

    * [Run the scenario with a custom policy](../notebooks/policy.md)