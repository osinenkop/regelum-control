## Why We Need Configs

In machine learning model development, especially in complex areas like reinforcement learning and control systems, navigating through numerous parameters and elements such as scenarios, policies, and simulators can be challenging.

Configuration files mitigate this complexity by externalizing parameters and component definitions, thereby:

- Enhancing organization by centralizing parameter management.
- Enabling reusability and sharing of settings without code duplication.
- Increasing flexibility, allowing parameter adjustments and component swaps without altering the core code, which supports quick experimentation.

At Regelum, we utilize configuration files through our adaptation of the Hydra framework, called ReHydra. This version adds extra conveniences for object instantiation and component interlinking.

Configuration files are particularly useful for projects with multiple entities and for those needing rapid testing of various hypotheses linked to hyperparameters. Hydra also simplifies hyperparameter grid searches.

We list the code from our [policy tutorial](../notebooks/policy.md) below. We will provide step-by-step guide of how to transition that code into configuration files.

!!! example annotate "Full code listing of [policy tutorial](../notebooks/policy.md)" 
    ```python
    from regelum.policy import Policy
    from regelum.system import KinematicPoint
    import numpy as np
    from regelum.scenario import Scenario
    from regelum.simulator import CasADi
    from regelum.callback import (
        ScenarioStepLogger,
        HistoricalDataCallback,
    )
    from regelum import set_ipython_env
    from regelum import callback
    import matplotlib.pyplot as plt

    # by default animation callbacks are attached to the system so we detach them
    KinematicPoint = callback.detach(KinematicPoint)

    callbacks = [ScenarioStepLogger, HistoricalDataCallback]
    ScenarioStepLogger.cooldown = 0.1
    callbacks = set_ipython_env(callbacks=callbacks, interactive=True)


    class SimplePolicy(Policy):
        def __init__(self, gain):
            super().__init__()
            self.gain = gain

        def get_action(self, observation):
            return -self.gain * observation
    
    policy = SimplePolicy(gain=0.2)
    system = KinematicPoint()
    
    sampling_time = 0.01
    simulator = CasADi(
        system=system,
        state_init=np.array([-10, 10]),
        max_step=sampling_time / 10,
    )

    scenario = Scenario(
        policy=SimplePolicy(gain=0.2),
        simulator=simulator,
        sampling_time=sampling_time,
        N_episodes=1,
        N_iterations=1,
    )

    scenario.run()
    ```

In the end of the tutorial we will establish a project with the following structure:
```
project
├── presets
│   ├── main.yaml
│   ├── common
│   │   └── common.yaml
│   ├── policy
│   │   └── simple_policy.yaml
│   ├── scenario
│   │   └── scenario.yaml
│   ├── simulator
│   │   └── casadi.yaml
│   └── system
│       └── kin_point.yaml
├── src
│   ├── __init__.py
│   └── policy.py
└── run.py
```
The final source code for this tutorial is avaliable [here](https://github.com/osinenkop/regelum-control/tree/master/docs/rgproject)

## Configuring policy

Let us transfer the code of `SimplePolicy` class to `src/` folder.

!!! example annotate "`project/src/__init__.py`" 
    ```python
    from . import policy
    ```

!!! example annotate "`project/src/policy.py`" 
    ```python
    from regelum.policy import Policy

    class SimplePolicy(Policy):
        def __init__(self, gain: float):
            super().__init__()
            self.gain = gain

        def get_action(self, observation):
            return -self.gain * observation

    ```

### Step 1: Create the `SimplePolicy` Config File

We will create a configuration file for the `SimplePolicy`. This config will be stored in the `policy` directory within the `presets` folder, based on the structure provided.

!!! example annotate "`presets/policy/simple_policy.yaml`" 
    ```yaml
    # Qualified import path for the SimplePolicy class
    _target_: src.policy.SimplePolicy
    # Constructor argument
    gain: 0.2                         
    ```

### Step 2: Include the Policy Config in the Main Config

Next, your main configuration file `main.yaml` will include the policy predefined configuration.

!!! example annotate "`presets/main.yaml`"
    ```yaml
    defaults:
    - policy: simple_policy  # Includes our SimplePolicy definition
    # ...other default configurations
    ```
??? example annotate "Full code listing of `presets/main.yaml`"
    ```yaml
    # This is the main preset file for the project
    # Here we define the presets we want to use

    defaults:
    - policy: simple_policy
    - system: kin_point 
    - simulator: casadi
    - scenario: scenario
    - common: common

    # below we can define our callbacks we want to use in 
    # our agent environment loop
    callbacks:
    - regelum.callback.ScenarioStepLogger
    - regelum.callback.HistoricalDataCallback


    # let us define the outputs folder for our runs
    rehydra:
    sweep:
        dir: ${oc.env:REGELUM_DATA_DIR}/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
    ```

## Configuring system

### Step 1: Define Your System Configuration

For `KinematicPoint` system, we will create a config file named `kin_point.yaml` in the `presets/system` directory of your project:

!!! example annotate "`presets/system/kin_point.yaml`"
    ```yaml
    _target_: regelum.system.KinematicPoint
    ```

The Kinematic Point system is designed without constructor keyword arguments (kwargs), resulting in the presets/system/kin_point.yaml file containing only a single line. This line 
```yaml
_target_: regelum.system.KinematicPoint
```
simply specifies the class that needs to be instantiated.

### Step 2: Reference System Config in the Main Config

Your `main.yaml` file should include the system configuration file to stitch together different parts of your project.

!!! example annotate "`presets/main.yaml`"
    ```yaml
    defaults:
    - policy: simple_policy
    - system: kin_point 
    # More component configurations go here like policy, simulator, scenario, etc.
    ```

??? example annotate "Full code listing of `presets/main.yaml`"
    ```yaml
    # This is the main preset file for the project
    # Here we define the presets we want to use

    defaults:
    - policy: simple_policy
    - system: kin_point 
    - simulator: casadi
    - scenario: scenario
    - common: common

    # below we can define our callbacks we want to use in 
    # our agent environment loop
    callbacks:
    - regelum.callback.ScenarioStepLogger
    - regelum.callback.HistoricalDataCallback


    # let us define the outputs folder for our runs
    rehydra:
    sweep:
        dir: ${oc.env:REGELUM_DATA_DIR}/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
    ```

## Create `common.yaml` 

The sampling_time parameter is utilized as a constructor argument in both the simulator and the scenario. Therefore, let's create a separate configuration file for sampling_time for further usage.

!!! example annotate "`presets/common/common.yaml`"
    ```yaml
    sampling_time: 0.01
    ```


## Configuring simulator

### Step 1: Craft Your Simulator Configuration

Create a configuration file for your simulator within the `presets/simulator` directory. In this file, you can specify all the parameters relevant to the simulator you are working with. Let's say we're setting up a CasADi-based simulator:

!!! example annotate "`presets/simulator/casadi.yaml`"
    ```yaml
    _target_: regelum.simulator.CasADi

    # The "%%" syntax is used for defining fields that aren't direct constructor arguments but are used elsewhere
    x%%: -10
    y%%: -10

    # Inline instantiation of a numpy array using the "=" syntax sugar
    state_init: = numpy.array([${.x%%}, ${.y%%}]) 

    # The tilde "~" symbol is used to indicate that the "system" will be instantiated dynamically from another config
    system: ~ system

    # Inline initialization of max_step kwarg using the "=" syntax sugar
    max_step: = ${common.sampling_time} / 10
    ```

Here's a breakdown of some of the syntax sugar used in ReHydra configs:

- **Inline Numpy Array Instantiation**: The `=` symbol allows for inline code execution within the config file. In the context of Regelum, you can directly instantiate objects like numpy arrays without having to execute additional Python code outside the config file.

- **Tilde for Dynamic Instantiation**: The tilde (`~`) symbol enables dynamic instantiation of objects from other configs. This feature provides a clean and concise way to compose your setup from modular components defined in other configuration files.

- **Double Percent Sign (`%%`) for Non-Constructor Fields**: The `%%` symbol is used to specify fields in the config that are not direct constructor arguments but are still required for other operations or configurations. This feature effectively separates parameters that are used purely within the configuration environment from those that are passed directly to object constructors.

- **Dollar Sign (`$`) for Dynamic Substitution**: ReHydra allows you to use the `$` syntax to reference parameters defined in other config files, enabling a single source of truth for parameters that are shared across multiple components.

### Step 2: Include Simulator Config in the Main Config

The main configuration file will now reference this simulator config to compose the larger system.

!!! example annotate "`presets/simulator/main.yaml`"
    ```yaml
    defaults:
    - policy: simple_policy
    - system: kin_point 
    - simulator: casadi
    # Other component configurations
    ```

### Advantages of Using Syntax Sugar

- **Clarity and Compactness**: With inline instantiation and the tilde symbol for referencing other configs, your main config file remains clear and concise, focused only on the high-level structure of your project.

- **Flexibility and Extensibility**: Non-constructor parameters are clearly delineated, making it easier to manage and override settings for different use cases or experiments.

- **Reduced Boilerplate**: Direct instantiation of objects like numpy arrays within the config minimizes the need for boilerplate code, helping maintain a focus on the logic that matters.

- **Seamless Component Composition**: By dynamically linking components using the tilde symbol, you encourage the creation of reusable, modular configurations that can be easily shared and altered without affecting the core application logic.

## Configuring scenario 

### Step 1: Define Your Scenario Configuration

Your scenario configuration specifies the setting in which your agent interacts and learns. This includes references to the agent's policy, the simulator with the system being controlled, and parameters such as the sampling time or number of episodes. Start by creating a file for the scenario configuration in the `presets/scenario` directory:

!!! example annotate "`presets/scenario/scenario.yaml`"
    ```yaml
    _target_: regelum.scenario.Scenario

    # Referencing common parameters using the "$" symbol for dynamic substitutions
    sampling_time: $ common.sampling_time
    N_episodes: 1
    N_iterations: 1

    # Referencing the policy and simulator through the tilde syntax for dynamically instantiating other configs
    policy: ~ policy
    simulator: ~ simulator
    ```

In this config file:

- **Dollar Sign (`$`) for Dynamic Substitution**: ReHydra allows you to use the `$` syntax to reference parameters defined in other config files, enabling a single source of truth for parameters that are shared across multiple components.
- **Tilde (`~`) for Dynamic Instantiation**: Like with the simulator configuration, the scenario config uses the `~` symbol to point to other configs for policy and simulator instantiations without directly embedding their specifications, promoting modularity.

### Step 2: Incorporate the Scenario Config into the Main Config

You must include the scenario configuration in the `main.yaml` file to ensure it's picked up when the application runs.

!!! example annotate "`presets/main.yaml`"
    ```yaml
    defaults:
    - policy: simple_policy
    - system: kin_point 
    - simulator: casadi
    - scenario: scenario
    ```

## Wrap it up in `main.yaml`

The `main.yaml` file serves as the central configuration file in a project using the Regelum framework with ReHydra. It acts as an entry point for Hydra to construct the overall configuration by aggregating component configurations defined in separate files. Let's walk through a sample `main.yaml` file, explaining the purpose of each line and section:

!!! example annotate "`presets/main.yaml`"
    ```yaml
    # 'defaults' lists all the component configurations that should be included by default when this file is used.
    defaults:
    - system: kin_point  # Includes the system configuration from 'presets/system/kin_point.yaml'.
    - policy: simple_policy  # Includes the policy configuration from 'presets/policy/simple_policy.yaml'.
    - simulator: casadi  # Includes the simulator configuration from 'presets/simulator/casadi.yaml'.
    - scenario: scenario  # Includes the scenario configuration from 'presets/scenario/scenario.yaml'.
    - common: common  # Includes common settings from 'presets/common/common.yaml'.

    # 'callbacks' defines a list of callback classes that will be used in the environment loop of the agent.
    callbacks:
    - regelum.callback.ScenarioStepLogger  # Logs steps of the scenario, useful for debugging or monitoring progress.
    - regelum.callback.HistoricalDataCallback  # Stores historical data throughout the simulation, which can be useful for analysis or visualization.

    # 'rehydra' is a section specific to ReHydra, allowing for additional customizations, such as where to store output files.
    rehydra:
    sweep:
        dir: ${oc.env:REGELUM_DATA_DIR}/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}  # Defines the directory path for sweep outputs.
    ```

### Explanation of each section and line:

- **`defaults`**: This section lists the default configurations that the Hydra framework should include when the application starts. Each item in the list corresponds to a specific component of the RL system you're configuring.

  - `system: kin_point`: Specifies that the default system configuration is found in the `kin_point.yaml` file within the `presets/system` directory. This defines the dynamic system or environment that will be simulated.
  - `policy: simple_policy`: Points to the policy settings defined in `simple_policy.yaml` located in the `presets/policy` directory. The policy determines how the agent behaves and makes decisions.
  - `simulator: casadi`: Includes the simulator's configuration from `casadi.yaml` in `presets/simulator`, which sets up how the system is simulated, for example using the CasADi toolkit.
  - `scenario: scenario`: Links to the scenario configuration from `scenario.yaml` in `presets/scenario`. The scenario orchestrates the interactions between the policy, simulator, and system over time.
  - `common: common`: Brings in common parameters that are shared across various components from the `common.yaml` file in the `presets/common` directory.

- **`callbacks`**: Defines a list of callback classes to be used during the execution of the simulation. Callbacks can perform various functions, such as logging information after each simulation step or collecting historical data for subsequent analysis.

  - [`regelum.callback.ScenarioStepLogger`][regelum.callback.ScenarioStepLogger]: A class that logs detailed information at each step of the scenario for monitoring and debugging purposes.
  - [`regelum.callback.HistoricalDataCallback`][regelum.callback.HistoricalDataCallback]: This callback collects and stores historical data throughout the execution of the scenario. It's typically used for post-analysis or real-time data visualization.

- **`rehydra`**: A custom section for additional configurations specific to ReHydra's extended functionalities.

  - `sweep.dir`: Specifies the directory where output files should be stored. It uses environment variable substitution (`${oc.env:REGELUM_DATA_DIR}`) to fetch the base directory from your environment and then appends a folder path that includes the current date and time (`${now:%Y-%m-%d}/${now:%H-%M-%S}`). This is particularly useful for organizing outputs from multiple runs or hyperparameter sweeps.


The `main.yaml` file is pivotal in ensuring that your project has a modular and coherent configuration setup. The system, policy, simulator, and scenario configurations are integrated here, providing a high-level overview of the experiment's settings. The callbacks and output directories are also specified, giving you a full picture of how the experiment will run and where the results will be saved. This structure lends itself to a clean and organized project, where changes to any part of the system can be made quickly and reliably without affecting other components.

## `run.py` 

The `run.py` file is the entry point script of a Regelum project configured with ReHydra. It initializes and executes the scenario as defined by the aggregated configuration stemming from `main.yaml`. Let's dive into the structure and functionality of this script:

!!! example annotate "`run.py`"
    ```yaml
    import regelum as rg

    @rg.main(config_path="presets", config_name="main")
    def main(cfg):
        scenario = ~cfg.scenario
        scenario.run()

    if __name__ == "__main__":
        main()
    ```

### Functionality of `run.py`:

- **Import Statement**: We start by importing `regelum`, which is our primary library that encapsulates the functionality for reinforcement learning and control theory research.

- **Main Function `@rg.main` Decorator**: This decorator provided by Regelum sets up the execution context for the script. It takes two arguments:
  - `config_path`: Specifies the directory where the Hydra configuration files are located, which in this case is `"presets"`. It lets the Hydra framework know where to find the relevant configurations to build the application context.
  - `config_name`: Determines the name of the primary configuration file to use, which in this setup is `"main"`. This file (`main.yaml`) integrates all the component configurations and acts as the launching pad for the project.

- **Main Function Definition**: The `main` function is defined as the central function that will be executed when the script runs. It takes a single argument `cfg`, which is the aggregated configuration object constructed by Hydra based on `main.yaml` and the associated component config files.

- **Scenario Instantiation and Execution**:
  - `scenario`: Here, the `~cfg.scenario` dynamically creates an instance of the scenario defined in the configuration files. It assembles the scenario by pulling in the policy, simulator, and other related settings as specified.
  - `scenario.run()`: This method call starts the execution of the scenario, running the series of episodes or steps as configured in the scenario parameters.

- **Main Guard**: The `if __name__ == "__main__":` guard ensures that the `main` function is called only if the script is executed directly, as opposed to being imported as a module in another script.

### `regelum_data`

When launched, the `run.py` script, along with Regelum and ReHydra, will create a `regelum_data` folder, unless it already exists. This directory is designated for storing operational data and artifacts generated during the runs. Inside `regelum_data`, there are two important subfolders:

- `regelum_data/mlruns`: This folder is used by MLflow for logging metrics, parameters, and artifacts associated with the different runs. By executing the `mlflow ui` command in the terminal inside `regelum_data` folder, users can start a web interface that provides a detailed and interactive view of all the run data logged by MLflow. This includes comparative analysis of different runs, visualization of metrics over time, and access to any saved artifacts. More details can be found on [official documentation website](https://mlflow.org/docs/latest/index.html).

- `regelum_data/mlruns`: This directory contains 'raw' data as well as other artifacts that do not fit into the structured logs of MLflow but are nonetheless important. This could be logs, model checkpoints, figures, or any other data that is captured during the execution of the scenario.
