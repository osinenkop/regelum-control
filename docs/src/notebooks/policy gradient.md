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


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:47] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.00</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.06</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.23</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.31</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.2</span> <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.5000</span>,  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:48] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7.55</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.06</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.46</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.06</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.46</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.16</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.15</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.31</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.24</span><span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">19.7559</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.4000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">48.6</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:49] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7.90</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.11</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.11</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.09</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.15</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.45</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.18</span><span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">36.9753</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.9000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">98.6</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:50] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7.48</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.58</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.58</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.3</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.27</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.39</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.17</span><span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">19.4675</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.3000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">47.1</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:51] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8.83</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.02</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.29</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.02</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.29</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.06</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.28</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.27</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.23</span><span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">39.2343</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.8000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">97.1</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:52] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.05</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.66</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.66</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.12</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.11</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.12</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.28</span><span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14.4455</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.7000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">38.6</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:53] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7.75</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.09</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.31</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.09</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.31</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.05</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.16</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.29</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.3</span> <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">33.4673</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.9000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">84.3</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:55] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">20.00</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:   <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.08</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.24</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.86</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.0000</span>,  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:56] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12.82</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.03</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.46</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:   <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.03</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.46</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.12</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.2</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.88</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">53.3908</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.4000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">48.6</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:57] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11.30</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.05</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.19</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:   <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.05</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.19</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.14</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.19</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.89</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">91.7726</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.9000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">98.6</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:58] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14.17</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.04</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.57</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:   <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.04</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.57</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.21</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.89</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">52.5646</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.2000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">45.7</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:44:59] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11.28</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.02</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.29</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:   <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.02</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.29</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.22</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.88</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">85.1233</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.7000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">95.7</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:45:00] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">15.26</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.06</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.63</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:   <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.06</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.63</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.16</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.2</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-9</span>.    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">50.9497</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.1000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">44.3</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:45:01] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11.87</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.06</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.26</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:   <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.06</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.26</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.17</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.2</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.81</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">88.4372</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.6000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">94.3</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:45:04] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.00</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-9.19</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9.38</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.93</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.5000</span>,  <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                           <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:45:05] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.00</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-9.24</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9.43</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.91</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16.4779</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.5000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">50.0</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:45:06] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.00</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-9.28</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9.52</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.76</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">29.4812</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.9000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">98.6</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:45:07] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.00</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-9.24</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9.55</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.92</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">15.6404</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.3000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">47.1</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:45:08] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.00</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-9.44</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9.57</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.91</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">29.1301</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.8000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">97.1</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[03:45:09] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> runn. objective: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.00</span>, state est.: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, observation:    <a href="file:///mnt/md0/rcognita/regelum/callback.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">callback.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///mnt/md0/rcognita/regelum/callback.py#1095" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1095</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.99</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.01</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>.   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.79</span><span style="font-weight: bold">]</span>, action: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-9.31</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9.62</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-8.98</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>.  <span style="font-weight: bold">]</span>, value: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14.7937</span>, <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>         time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3.1000</span> <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">44.3</span>%<span style="font-weight: bold">)</span>, episode: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, iteration: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span>                          <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                </span>
</pre>




    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 scenario.run()


    File /mnt/md0/rcognita/regelum/scenario.py:473, in RLScenario.run(self)
        471 def run(self):
        472     if not self.is_parallel:
    --> 473         return super().run()
        475     for self.iteration_counter in range(1, self.N_iterations + 1):
        476         # self.policy.model.stds *= self.annealing_exploration_factor
        477         one_episode_rl_scenarios = self.instantiate_rl_scenarios()


    File /mnt/md0/rcognita/regelum/scenario.py:140, in Scenario.run(self)
        138 for iteration_counter in range(1, self.N_iterations + 1):
        139     for episode_counter in range(1, self.N_episodes + 1):
    --> 140         self.run_episode(
        141             episode_counter=episode_counter, iteration_counter=iteration_counter
        142         )
        143         self.reload_scenario()
        145     self.reset_iteration()


    File /mnt/md0/rcognita/regelum/scenario.py:410, in RLScenario.run_episode(self, episode_counter, iteration_counter)
        409 def run_episode(self, episode_counter, iteration_counter):
    --> 410     super().run_episode(
        411         episode_counter=episode_counter, iteration_counter=iteration_counter
        412     )
        413     return self.data_buffer


    File /mnt/md0/rcognita/regelum/scenario.py:156, in Scenario.run_episode(self, episode_counter, iteration_counter)
        154 self.iteration_counter = iteration_counter
        155 while self.sim_status != "episode_ended":
    --> 156     self.sim_status = self.step()


    File /mnt/md0/rcognita/regelum/scenario.py:190, in Scenario.step(self)
        185     self.substitute_constraint_parameters(**self.constraint_parameters)
        186 estimated_state = self.observer.get_state_estimation(
        187     self.time, self.observation, self.action
        188 )
    --> 190 self.action = self.compute_action_sampled(
        191     self.time,
        192     estimated_state,
        193     self.observation,
        194 )
        195 self.simulator.receive_action(self.action)
        196 self.is_episode_ended = self.simulator.do_sim_step() == -1


    File /mnt/md0/rcognita/regelum/__internal/base.py:30, in apply_callbacks.__call__.<locals>.new_method(self2, *args, **kwargs)
         29 def new_method(self2, *args, **kwargs):
    ---> 30     res = method(self2, *args, **kwargs)
         31     if self.callbacks is None:
         32         try:


    Cell In[6], line 81, in GameScenario.compute_action_sampled(self, time, estimated_state, observation)
         79 @apply_callbacks()
         80 def compute_action_sampled(self, time, estimated_state, observation):
    ---> 81     return super().compute_action_sampled(time, estimated_state, observation)


    File /mnt/md0/rcognita/regelum/scenario.py:244, in Scenario.compute_action_sampled(self, time, estimated_state, observation)
        236 self.on_observation_received(time, estimated_state, observation)
        237 action = self.simulator.system.apply_action_bounds(
        238     self.compute_action(
        239         time=time,
       (...)
        242     )
        243 )
    --> 244 self.post_compute_action(observation, estimated_state)
        245 self.step_counter += 1
        246 self.action_old = action


    File /mnt/md0/rcognita/regelum/__internal/base.py:37, in apply_callbacks.__call__.<locals>.new_method(self2, *args, **kwargs)
         35         callbacks = []
         36 for callback in callbacks:
    ---> 37     callback(obj=self2, method=method.__name__, output=res)
         38 return res


    File /mnt/md0/rcognita/regelum/callback.py:216, in Callback.__call__(self, obj, method, output)
        214 try:
        215     if self.is_target_event(obj, method, output, triggers):
    --> 216         self.on_function_call(obj, method, output)
        217         done = True
        218         for base in self.__class__.__bases__:


    File /mnt/md0/rcognita/regelum/callback.py:1195, in HistoricalDataCallback.on_function_call(self, obj, method, output)
       1188     self.state_components_naming = (
       1189         [f"state_{i + 1}" for i in range(obj.simulator.system.dim_state)]
       1190         if obj.simulator.system.state_naming is None
       1191         else obj.simulator.system.state_naming
       1192     )
       1194 if method == "post_compute_action":
    -> 1195     self.add_datum(
       1196         {
       1197             **{
       1198                 "time": output["time"],
       1199                 "running_objective": output["running_objective"],
       1200                 "current_value": output["current_value"],
       1201                 "episode_id": output["episode_id"],
       1202                 "iteration_id": output["iteration_id"],
       1203             },
       1204             **dict(zip(self.action_components_naming, output["action"][0])),
       1205             **dict(
       1206                 zip(self.state_components_naming, output["estimated_state"][0])
       1207             ),
       1208             # **dict(
       1209             #     zip(self.state_components_naming, output["estimated_state"][0])
       1210             # ),
       1211         }
       1212     )
       1213 elif method == "dump_data_buffer":
       1214     _, data_buffer = output


    File /mnt/md0/rcognita/regelum/callback.py:820, in HistoricalCallback.add_datum(self, datum)
        818     self.data = pd.DataFrame({key: [value] for key, value in datum.items()})
        819 else:
    --> 820     self.data.loc[len(self.data)] = datum


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/core/indexing.py:716, in _LocationIndexer.__setitem__(self, key, value)
        713 self._has_valid_setitem_indexer(key)
        715 iloc = self if self.name == "iloc" else self.obj.iloc
    --> 716 iloc._setitem_with_indexer(indexer, value, self.name)


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/core/indexing.py:1682, in _iLocIndexer._setitem_with_indexer(self, indexer, value, name)
       1679     indexer, missing = convert_missing_indexer(indexer)
       1681     if missing:
    -> 1682         self._setitem_with_indexer_missing(indexer, value)
       1683         return
       1685 # align and set the values


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/core/indexing.py:2013, in _iLocIndexer._setitem_with_indexer_missing(self, indexer, value)
       2011     self.obj._mgr = df._mgr
       2012 else:
    -> 2013     self.obj._mgr = self.obj._append(value)._mgr
       2014 self.obj._maybe_update_cacher(clear=True)


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/core/frame.py:9066, in DataFrame._append(self, other, ignore_index, verify_integrity, sort)
       9063 else:
       9064     to_concat = [self, other]
    -> 9066 result = concat(
       9067     to_concat,
       9068     ignore_index=ignore_index,
       9069     verify_integrity=verify_integrity,
       9070     sort=sort,
       9071 )
       9072 if (
       9073     combined_columns is not None
       9074     and not sort
       (...)
       9079     # combined_columns.equals check is necessary for preserving dtype
       9080     #  in test_crosstab_normalize
       9081     result = result.reindex(combined_columns, axis=1)


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/util/_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        305 if len(args) > num_allow_args:
        306     warnings.warn(
        307         msg.format(arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(*args, **kwargs)


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/core/reshape/concat.py:359, in concat(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort, copy)
        155 """
        156 Concatenate pandas objects along a particular axis with optional set logic
        157 along the other axes.
       (...)
        344 ValueError: Indexes have overlapping values: ['a']
        345 """
        346 op = _Concatenator(
        347     objs,
        348     axis=axis,
       (...)
        356     sort=sort,
        357 )
    --> 359 return op.get_result()


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/core/reshape/concat.py:592, in _Concatenator.get_result(self)
        588             indexers[ax] = obj_labels.get_indexer(new_labels)
        590     mgrs_indexers.append((obj._mgr, indexers))
    --> 592 new_data = concatenate_managers(
        593     mgrs_indexers, self.new_axes, concat_axis=self.bm_axis, copy=self.copy
        594 )
        595 if not self.copy:
        596     new_data._consolidate_inplace()


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/core/internals/concat.py:251, in concatenate_managers(mgrs_indexers, axes, concat_axis, copy)
        247         b = new_block_2d(values, placement=placement)
        249     blocks.append(b)
    --> 251 return BlockManager(tuple(blocks), axes)


    File ~/.pyenv/versions/3.11.4/envs/regelum/lib/python3.11/site-packages/pandas/core/internals/managers.py:919, in BlockManager.__init__(self, blocks, axes, verify_integrity)
        914 ndim = 2
        916 # ----------------------------------------------------------------
        917 # Constructors
    --> 919 def __init__(
        920     self,
        921     blocks: Sequence[Block],
        922     axes: Sequence[Index],
        923     verify_integrity: bool = True,
        924 ):
        926     if verify_integrity:
        927         # Assertion disabled for performance
        928         # assert all(isinstance(x, Index) for x in axes)
        930         for block in blocks:


    KeyboardInterrupt: 



```python

```
