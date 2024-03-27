![image](https://github.com/osinenkop/regelum-control/blob/master/docs/src/gfx/regelum_full_logo.png)

# About

`Regelum-control` stands as a framework designed to address optimal control and reinforcement learning (RL) tasks within continuous-time dynamical systems. It is made for researchers and engineers in reinforcement learning and control theory.

A detailed documentation is available [here](https://regelum.aidynamic.io/).

# Features

- __Run pre-configured regelum algorithms with ease__. Regelum offers a set of implemented, ready-to-use algorithms in the domain of RL and Optimal Control. 
It provides flexibility through multiple optimization backends, including CasADi and PyTorch, to accommodate various computational needs.

-  __Stabilize your dynamical system with Regelum__. Regelum stands as a framework 
designed to address optimal control and reinforcement learning (RL) 
tasks within continuous-time dynamical systems. 
It comes equipped with an array of default systems accessible [here](#), 
alongside a detailed tutorial that provides clear instructions 
for users to instantiate their own environments.

- __Manage your experiment data__. Regelum seamlessly captures
every detail of your experiment with little to no configuration required. 
From parameters to performance metrics, every datum is recorded. Through integration with [MLflow](https://mlflow.org/), 
Regelum streamlines tracking, comparison and real-time monitoring of metrics.

-  __Reproduce your experiments with ease__. Commit hashes and diffs for every experiment are also stored in Regelum, 
offering the ability to reproduce your experiments at any time with simple terminal commands.

-  __Configure your experiments efficiently__. Our [Hydra](https://hydra.cc/) fork within Regelum introduces enhanced functionaly, 
making the configuration of your RL and Optimal Control tasks more accessible and user-friendly.

-  __Fine-tune your models to perfection__ and achieve peak performance with minimal effort. 
By integrating with Hydra, regelum inherently adopts Hydra's powerful hyperparameter tuning capability.

# Install regelum-control with pip

```bash
pip install regelum-control
```

# Licence

This project is licensed under the terms of the [MIT license](./LICENSE).

## Bibtex reference

```
@misc{regelum2024,
author =   {Pavel Osinenko, Grigory Yaremenko, Georgiy Malaniya, Anton Bolychev},
title =    {Regelum: a framework for simulation, control and reinforcement learning},
howpublished = {\url{https://github.com/osinekop/regelum-control}},
year = {2024}
}
```
