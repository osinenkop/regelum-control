# Mastering policy gradients with Regelum

For theoretical background we kindly refer a reader to our documentation (ToDo link).

In this tutorial we will implement two well-known policy gradient algorithms: REINFORCE and Vanilla Policy Gradient.

## REINFORCE.

Our plan is to implement:

1. Stochastic policy
    1. Stochastic model for action sampling
    2. Respective Policy with optimization procedure
    3. REINFORCE objective with baselines
2. Proper scenario and other basic entities needed to construct a pipeline
3. Define suitable callbacks to track our learning progress


```python
from regelum.optimizable.core.configs import TorchOptimizerConfig
from regelum.__internal.base import apply_callbacks
from regelum.policy import Policy
from regelum.scenario import RLScenario
from regelum.data_buffers import DataBuffer
from regelum.callback import (
    Callback,
    HistoricalCallback,
    ScenarioStepLogger,
    HistoricalDataCallback,
)
from regelum.system import ComposedSystem, KinematicPoint, ThreeWheeledRobotKinematic
from regelum.objective import RunningObjective
from regelum.critic import CriticTrivial
from regelum.model import (
    PerceptronWithTruncatedNormalNoise,
    ModelQuadLin,
    MultiplyByConstant,
)
from regelum.utils import rg
from regelum.simulator import CasADi
from regelum.event import Event
from regelum import set_ipython_env
from regelum.data_buffers.batch_sampler import RollingBatchSampler
from typing import List


import torch as th
import numpy as np

%matplotlib inline
```


```python
callbacks = [ScenarioStepLogger, HistoricalDataCallback]
ScenarioStepLogger.cooldown = 1
callbacks = set_ipython_env(callbacks=callbacks, interactive=True)
```


```python
from casadi.casadi import MX


def circle_bound(point):
    return point[0] ** 2 + point[1] ** 2 - 9


class KinematicPointRestricted(KinematicPoint):

    def compute_state_dynamics(self, time, state, inputs, _native_dim: bool = False):
        Dstate = super().compute_state_dynamics(time, state, inputs, _native_dim)
        return rg.if_else(
            circle_bound(state) <= -1e-7,
            Dstate,
            rg.if_else(
                rg.dot(Dstate, state) < 0,
                Dstate,
                rg.zeros(Dstate.shape[1], prototype=Dstate),
            ),
        )


class ThreeWheeledRobotKinematicRestricted(ThreeWheeledRobotKinematic):
    def compute_state_dynamics(self, time, state, inputs, _native_dim: bool = False):
        Dstate = super().compute_state_dynamics(time, state, inputs, _native_dim)
        return rg.if_else(
            circle_bound(state[:2]) <= -1e-7,
            Dstate,
            rg.if_else(
                rg.dot(Dstate[:2], state[:2]) < 0,
                Dstate,
                rg.zeros(Dstate.shape[1], prototype=Dstate),
            ),
        )


homicidal_chauffeur_system = ComposedSystem(
    sys_left=KinematicPointRestricted(),
    sys_right=ThreeWheeledRobotKinematicRestricted(),
    output_mode="both",
    io_mapping=[],  # means that no output of the left system is used as input to the right one.
    action_bounds=KinematicPoint._action_bounds
    + ThreeWheeledRobotKinematic._action_bounds,
)  # This concatenated system perfectly suits for homicidal chauffeur modelling
```


```python
def reinforce_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    observations: th.Tensor,
    actions: th.Tensor,
    tail_values: th.Tensor,
    baselines: th.Tensor,
    N_episodes: int,
) -> th.FloatTensor:
    log_pdfs = policy_model.log_pdf(observations, actions)
    target_objectives = tail_values
    target_objectives -= baselines

    return (log_pdfs * target_objectives).sum() / N_episodes
```


```python
class JointPolicy(Policy):
    def __init__(
        self,
        pedestrian_model: PerceptronWithTruncatedNormalNoise,
        chauffeur_model: PerceptronWithTruncatedNormalNoise,
        system: ComposedSystem,
    ):
        def freeze_stds(params):
            for p in params():
                if p[0] == "stds":
                    p[1].requires_grad_(False)
            return params

        iter_batches_kwargs = {
            "batch_sampler": RollingBatchSampler,
            "dtype": th.FloatTensor,
            "mode": "full",
            "n_batches": 1,
            "device": "cpu",
        }
        super().__init__(
            system=system,
            optimizer_config=TorchOptimizerConfig(
                n_epochs=300,
                data_buffer_iter_bathes_kwargs=iter_batches_kwargs,
                opt_method_kwargs=dict(lr=1e-3),
            ),
            action_bounds=system.sys_left.action_bounds
            + system.sys_right.action_bounds,
        )
        self.pedestrian_model = pedestrian_model
        self.chauffeur_model = chauffeur_model
        self.model_to_optimize = self.pedestrian_model
        self.next_baseline = 0.0

        ## Define an optimization problem here

        self.pedestrian_model_weigths = self.create_variable(
            name="pedestrian_model_weights", like=self.pedestrian_model.named_parameters
        ).register_hook(freeze_stds)
        self.chauffeur_model_weights = self.create_variable(
            name="chauffeur_model_weights", like=self.chauffeur_model.named_parameters
        ).register_hook(freeze_stds)
        self.objective_inputs = [
            self.create_variable(name=variable_name, is_constant=True)
            for variable_name in self.data_buffer_objective_keys()
        ]
        self.register_objective(
            self.objective_function, variables=self.objective_inputs
        )

    def switch_model_to_optimize(self):
        self.model_to_optimize = (
            self.pedestrian_model
            if self.model_to_optimize == self.chauffeur_model
            else self.chauffeur_model
        )
        return self.model_to_optimize

    def action_col_idx(self):
        return (
            slice(0, self.system.sys_left.dim_inputs)
            if self.model_to_optimize == self.pedestrian_model
            else slice(self.system.sys_right.dim_inputs, None)
        )

    def objective_function(
        self,
        observation: th.Tensor,
        action: th.Tensor,
        tail_value: th.Tensor,
        baseline: th.Tensor,
    ):
        actions_of_current_model = action[:, self.action_col_idx()]
        return reinforce_objective(
            policy_model=self.model_to_optimize,
            N_episodes=self.N_episodes,
            observations=observation,
            actions=actions_of_current_model,
            tail_values=tail_value,
            baselines=baseline,
        )

    def calculate_last_values(self, data_buffer: DataBuffer) -> np.ndarray:
        data = data_buffer.to_pandas(
            keys=["episode_id", "current_value"]
        )  # Convert data_buffer to pandas
        data["episode_id"] = data["episode_id"].astype(int)  # Convert episode id to int
        data["current_value"] = data["current_value"].astype(
            float
        )  # Convert current value to float
        return (
            data.groupby("episode_id")[
                "current_value"
            ]  # Group by episode id and get the last value of each episode
            .last()
            .loc[data["episode_id"]]
            .values.reshape(-1)
        )

    def calculate_tail_values(
        self,
        data_buffer: DataBuffer,
    ) -> np.ndarray:
        data_buffer.update(
            {
                "value": self.calculate_last_values(data_buffer),
            }
        )
        data = data_buffer.to_pandas(keys=["episode_id", "current_value"])
        last_values = data_buffer.data["value"].astype(float)
        groupby_episode_values = data.groupby(["episode_id"])["current_value"]
        current_values_shifted = groupby_episode_values.shift(
            periods=1, fill_value=0.0
        ).values.reshape(-1)

        return last_values - current_values_shifted

    def calculate_baseline(self, data_buffer: DataBuffer) -> np.ndarray:
        baseline = self.next_baseline

        step_ids = (
            data_buffer.to_pandas(keys=["step_id"]).astype(int).values.reshape(-1)
        )
        self.next_baseline = (
            data_buffer.to_pandas(keys=["tail_value", "step_id"])
            .astype(float)
            .groupby("step_id")
            .mean()
            .loc[step_ids]
        ).values
        if isinstance(baseline, float):
            return np.full(shape=len(data_buffer), fill_value=baseline)
        else:
            return baseline

    def update_data_buffer(self, data_buffer: DataBuffer) -> None:
        data_buffer.update({"tail_value": self.calculate_tail_values(data_buffer)})
        data_buffer.update({"baseline": self.calculate_baseline(data_buffer)})

    def get_action(self, observation: np.array) -> np.array:
        action_pedestrian = self.pedestrian_model(th.FloatTensor(observation))
        action_chauffeur = self.chauffeur_model(th.FloatTensor(observation))
        action = rg.hstack((action_pedestrian, action_chauffeur)).detach().cpu().numpy()
        return action

    def data_buffer_objective_keys(self):
        return [
            "observation",
            "action",
            "tail_value",
            "baseline",
        ]

    def optimize(self, data_buffer: DataBuffer) -> None:
        self.N_episodes = len(np.unique(data_buffer.data["episode_id"]))
        self.update_data_buffer(data_buffer)
        opt_kwargs = data_buffer.get_optimization_kwargs(
            keys=self.data_buffer_objective_keys(),
            optimizer_config=self.optimizer_config,
        )
        super().optimize_tensor(**opt_kwargs)
```


```python
class PedestrialRunningObjectiveModel(ModelQuadLin):
    def __init__(self, weights=rg.array([10, 0, 0, 0, 0])):
        super().__init__(
            quad_matrix_type="diagonal",
            dim_inputs=4,
            weights=weights,
        )
        self.eps = 1.0e-5

    def __call__(self, *argin, **kwargs):
        observation = argin[0]
        pedestrian_pos = observation[0, :2]
        chauffeur_pos = observation[0, 2:-1]
        distance = rg.norm_2(pedestrian_pos - chauffeur_pos)
        inv_distance = rg.array(
            [
                [(1 / (distance + self.eps)) if distance < 2 else 0]
            ]  # We punish a pedestrian for close distance to the chauffeur
        )
        action = argin[1]
        return super().__call__(inv_distance, action, **kwargs)


class ChauffeurRunningObjectiveModel(ModelQuadLin):
    def __init__(self, weights=rg.array([10, 0, 0, 0, 0])):
        super().__init__(
            quad_matrix_type="diagonal",
            dim_inputs=4,
            weights=weights,
        )

    def __call__(self, *argin, **kwargs):
        observation = argin[0]
        pedestrian_pos = observation[0, :2]
        chauffeur_pos = observation[0, 2:-1]
        distance = rg.array([[rg.norm_2(pedestrian_pos - chauffeur_pos)]])
        action = argin[1]
        return super().__call__(distance, action, **kwargs)


class GameScenario(RLScenario):
    def __init__(
        self,
        policy: JointPolicy,
        system: ComposedSystem,
        pedestrial_running_objective_model: PedestrialRunningObjectiveModel,
        chauffeur_running_objective_model: ChauffeurRunningObjectiveModel,
        state_init: np.ndarray = rg.array([1.0, 1.0, 0.0, 0.0, np.pi / 4]),
    ):
        self.pedestrial_running_objective = RunningObjective(
            model=pedestrial_running_objective_model
        )
        self.chauffeur_running_objective = RunningObjective(
            model=chauffeur_running_objective_model
        )
        super().__init__(
            policy=policy,
            critic=CriticTrivial(),
            running_objective=self.pedestrial_running_objective,
            simulator=CasADi(
                system=system, state_init=state_init, time_final=7, max_step=0.01
            ),
            policy_optimization_event=Event.reset_iteration,
            discount_factor=0.95,
            sampling_time=0.1,
            N_episodes=10,
            N_iterations=200,
            is_parallel=False,
        )
        self.policy: JointPolicy

    def switch_running_objective(self):
        self.running_objective = (
            self.pedestrial_running_objective
            if self.running_objective == self.chauffeur_running_objective
            else self.chauffeur_running_objective
        )

    @apply_callbacks()
    def compute_action_sampled(self, time, estimated_state, observation):
        return super().compute_action_sampled(time, estimated_state, observation)

    def reset_iteration(self):
        super().reset_iteration()
        self.switch_running_objective()
        policy_weights_to_fix, policy_weights_to_unfix = (
            ["pedestrian_model_weights", "chauffeur_model_weights"]
            if isinstance(self.running_objective.model, ChauffeurRunningObjectiveModel)
            else ["chauffeur_model_weights", "pedestrian_model_weights"]
        )
        self.policy.fix_variables([policy_weights_to_fix])
        self.policy.unfix_variables([policy_weights_to_unfix])
        self.policy.switch_model_to_optimize()
```


```python
pedestrian_model = PerceptronWithTruncatedNormalNoise(
    dim_input=5,
    dim_output=2,
    dim_hidden=40,
    n_hidden_layers=3,
    hidden_activation=th.nn.Tanh(),
    output_bounds=homicidal_chauffeur_system.sys_left.action_bounds,
    is_truncated_to_output_bounds=True,
    output_activation=MultiplyByConstant(0.1),
    stds=[0.1] * 2,
)

chauffeur_model = PerceptronWithTruncatedNormalNoise(
    dim_input=5,
    dim_output=2,
    dim_hidden=40,
    n_hidden_layers=3,
    hidden_activation=th.nn.Tanh(),
    output_bounds=homicidal_chauffeur_system.sys_left.action_bounds,
    is_truncated_to_output_bounds=True,
    output_activation=MultiplyByConstant(0.1),
    stds=[0.1] * 2,
)


policy = JointPolicy(
    pedestrian_model=pedestrian_model,
    chauffeur_model=chauffeur_model,
    system=homicidal_chauffeur_system,
)

scenario = GameScenario(
    policy=policy,
    system=homicidal_chauffeur_system,
    pedestrial_running_objective_model=PedestrialRunningObjectiveModel(),
    chauffeur_running_objective_model=ChauffeurRunningObjectiveModel(),
)
```


```python
scenario.run()
```


```python
callbacks[1].data[
    (callbacks[1].data.episode_id == 8) & (callbacks[1].data.iteration_id == 11)
][[f"state_{i + 1}" for i in range(4)]].plot()
```


    
![png](data:image/png;base64, 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)
    



```python
callbacks[1].data[
    (callbacks[1].data.episode_id == 8) & (callbacks[1].data.iteration_id == 2)
][[f"state_{i + 1}" for i in range(4)]].plot()
```


    
![png](data:image/png;base64, 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)
    



```python

```
