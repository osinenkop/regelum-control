---
# title: MUUJUUJ
hide:
#   - title
  - navigation
#   - toc
---
# Regelum control

![3wrobot_mpc](https://regelum.aidynamic.group/gfx/3wrobot_mpc.gif){ width=33% }
![cartpole_lyapunov](https://regelum.aidynamic.group/gfx/cartpole_lyapunov.gif){  width=33% }
![pendulum_reinforce](https://regelum.aidynamic.group/gfx/pendulum_reinforce.gif){ width=33% }

## Overview
Regelum-control stands as a framework designed to address optimal control and reinforcement learning (RL) tasks within continuous-time dynamical systems. It is made for researchers and engineers in reinforcement learning and control theory. 


<div class="grid cards" markdown>

<!-- -   :octicons-light-bulb-16:{ .lg .middle } __Get to know Regelum__

    ---

    Dive into the introduction for an overview of its principal concepts and features.

    [:octicons-arrow-right-24: Introduction](introduction.md) -->

<!-- -   :octicons-rocket-16:{ .lg .middle } __Set it up__

    ---

    Install Regelum using pip and easily dive into your first simulation experience.


    [:octicons-arrow-right-24: Quick start][] -->

-   :octicons-book-16:{ .lg .middle } __Master Regelum__

    ---

    Take a tutorial on the fundamentals of Regelum and delve into its core concepts.

    [:octicons-arrow-right-24: Tutorial](tutorials/introduction)

-   :octicons-rocket-16:{ .lg .middle } __Try it out__

    ---

    Experience Regelum in action: implement the policy gradient method from scratch. 

    [:octicons-arrow-right-24: Try it out](notebooks/policy_gradient)

<!-- -   :octicons-light-bulb-16:{ .lg .middle } __Get to know Regelum__

    ---

    Dive into the introduction for an overview of its principal concepts and features.

    [:octicons-arrow-right-24: Introduction](introduction.md)

-   :octicons-rocket-16:{ .lg .middle } __Try it out__

    ---

    Experience Regelum in action: launch our interactive colab demo. 

    [:octicons-arrow-right-24: Try it in colab](#) -->

</div>

## Features

:octicons-cpu-16: __Run pre-configured regelum algorithms with ease__. Regelum offers a set of implemented, [ready-to-use algorithms](tutorials/stable-presets.md) in the domain of RL and Optimal Control. 
It provides flexibility through multiple optimization backends, including CasADi and PyTorch, to accommodate various computational needs.

:material-robot-industrial: __Stabilize your dynamical system with Regelum__. Regelum stands as a framework 
designed to address optimal control and reinforcement learning (RL) 
tasks within continuous-time dynamical systems. 
It comes equipped with an array of default systems accessible [here](systems/kin_point/), 
alongside a detailed tutorial that provides clear instructions 
for users to instantiate their own environments.

:simple-mlflow: __Manage your experiment data__. Regelum seamlessly captures
every detail of your experiment with little to no configuration required. 
From parameters to performance metrics, every datum is recorded. Through integration with [MLflow](https://mlflow.org/), 
Regelum streamlines tracking, comparison and real-time monitoring of metrics.

:fontawesome-solid-repeat: __Reproduce your experiments with ease__. Commit hashes and diffs for every experiment are also stored in Regelum, 
offering the ability to reproduce your experiments at any time with simple terminal commands.

:material-wrench-outline: __Configure your experiments efficiently__. Our [Hydra](https://hydra.cc/) fork within Regelum introduces enhanced functionaly, 
making the configuration of your RL and Optimal Control tasks more accessible and user-friendly.

:material-bullseye-arrow: __Fine-tune your models to perfection__ and achieve peak performance with minimal effort. 
By integrating with Hydra, regelum inherently adopts Hydra's powerful hyperparameter tuning capability.

## Install with pip
<!-- termynal -->

```
$ pip install regelum-control
---> 100%
Installed
```

## Licence

This project is licensed under the terms of the [MIT license](TODO).

