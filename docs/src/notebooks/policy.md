```python
from regelum.policy import Policy
from regelum.system import KinematicPoint
from regelum.utils import rg
from regelum.scenario import Scenario
from regelum.simulator import CasADi
from regelum.callback import (
    ScenarioStepLogger,
    HistoricalDataCallback,
)
from regelum import set_jupyter_env
%matplotlib inline
```


```python
callbacks = [ScenarioStepLogger, HistoricalDataCallback]
ScenarioStepLogger.cooldown = 0.1
callbacks = set_jupyter_env(callbacks=callbacks, interactive=True)
```


```python
class SimplePoplicy(Policy):
    def __init__(self, system=KinematicPoint(), gain=0.2):
        super().__init__(system=system)
        self.gain = gain

    def get_action(self, observation):
        return -self.gain * observation


policy = SimplePoplicy()
system = KinematicPoint()
```


```python
simulator = CasADi(
    system=system,
    state_init=rg.array([-10, 10]),
    action_init=rg.array([0]),
    max_step=1e-3,
    first_step=1e-6,
    atol=1e-5,
    rtol=1e-3,
)

scenario = Scenario(
    policy=policy,
    simulator=simulator,
    sampling_time=0.01,
    N_episodes=1,
    N_iterations=1,
)
```


```python
scenario.run()
```


```python
callbacks[1].plot(name="observations")
```
