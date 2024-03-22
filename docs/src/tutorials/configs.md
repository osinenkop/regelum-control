## Why We Need Configs

In the development of machine learning models, particularly in complex domains such as reinforcement learning and control systems, we often deal with multi-faceted settings involving numerous parameters and entities like scenarios, policies, simulators, and so forth.

Managing these components directly in the code becomes unwieldy as the complexity grows. This is where configuration files come in handy. By externalizing parameters and component definitions, configs offer:

- **Better organization**: Keeping parameters in a centralized location makes them easier to manage and review.
- **Reusability**: Configs allow you to reuse and share settings across experiments without duplicating code.
- **Flexibility**: You can tweak parameters and switch components without touching the core codebase, facilitating experimentation and rapid prototyping.

Regelum harnesses the power of configuration files by leveraging a modified version of the Hydra framework, which we refer to as ReHydra. This fork introduces additional syntax sugar to make the instantiation of objects and linking between them even more straightforward.


Configuring pipelines may be usefull in case if your project requires many entities to instantiate and if you want to fastly test different hypothesis which depend on the hyperrparameters. 
Moreover, Hydra provides easy grid search through hyperparameters.  

We provide the full code from [policy tutorial](../notebooks/policy.md) and we will show you the approach of how to transfer the code to configs.  


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

    system = KinematicPoint()

    simulator = CasADi(
        system=system,
        state_init=np.array([-10, 10]),
    )

    scenario = Scenario(
        policy=SimplePolicy(gain=0.2),
        simulator=simulator,
        sampling_time=0.01,
        N_episodes=1,
        N_iterations=1,
    )

    scenario.run()
    ```

As a result we will come up with the project of the following structure. 
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
We will cover all the content of the files in our project.


Firstly, note that we have some new code that defines policy:
```python

    class SimplePolicy(Policy):
        def __init__(self, gain):
            super().__init__()
            self.gain = gain

        def get_action(self, observation):
            return -self.gain * observation
```
so, that's why we provide below

!!! example annotate "Full code listing of `project/src/policy.py`" 
    ```python
    from regelum.policy import Policy

    class SimplePolicy(Policy):
        def __init__(self, gain: float):
            super().__init__()
            self.gain = gain

        def get_action(self, observation):
            return -self.gain * observation

    ```

!!! example annotate "Full code listing of **src/__init__.py**" 
    ```python
    from . import policy
    ```

### Step 1: Create the `SimplePolicy` Config File

We will create a configuration file for the `SimplePolicy`. This config will be stored in the `policy` directory within the `presets` folder, based on the structure provided.

!!! example annotate "Full code listing of **presets/policy/simple_policy.yaml**" 
    ```yaml
    # Qualified import path for the SimplePolicy class
    _target_: src.policy.SimplePolicy
    # Constructor argument
    gain: 0.2                         
    ```

### Step 2: Include the Policy Config in the Main Config

Next, your main configuration file `main.yaml` will include the policy predefined configuration.

!!! example annotate "Code listing of `presets/main.yaml` which defines policy"
    ```yaml
    defaults:
    - policy: simple_policy  # Includes our SimplePolicy definition
    # ...other default configurations
    ```

### Step 3: Instantiate the Policy in the `run.py` File

Aligning with the provided structure of your `run.py` file and using ReHydra with Regelum, we instantiate the policy as follows:

!!! example annotate "Code listing of `project/run.py` which defines policy"
    ```python
    import regelum as rg

    @rg.main(config_path="presets", config_name="main")
    def launch(cfg):
        policy = rg.hydra.utils.instantiate(cfg.policy)
        # Now 'policy' is an instance of SimplePolicy configured with 'gain' set to 0.2
        # You may now use 'policy' wherever needed in your application

    if __name__ == "__main__":
        job_results = launch()
    ```

In the `run.py` script:

- We use the `@rg.main` decorator from Regelum, which points to the folder `presets` where our configuration files reside.
- The `config_name="main"` argument tells ReHydra to use our main configuration file, which includes references to all component configurations, such as the policy.
- Within the `launch` function, we can then instantiate `SimplePolicy` using `rg.hydra.utils.instantiate(cfg.policy)`, which creates an instance of `SimplePolicy` with the parameters specified in `simple_policy.yaml`.
