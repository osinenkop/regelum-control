from regelum.scenario import PPO
from regelum.model import (
    PerceptronWithTruncatedNormalNoise,
    ModelQuadLin,
    ModelPerceptron,
)
from regelum.system import ThreeWheeledRobotNI
import torch
import regelum
import numpy as np
from regelum.simulator import CasADi
from regelum.objective import RunningObjective
from copy import deepcopy


@regelum.main(
    config_path="/mnt/md0/rcognita/presets/stable-presets", config_name="main"
)
def launch(cfg):
    # cfg = deepcopy(cfg)
    system = ThreeWheeledRobotNI()
    simulator = CasADi(system=system, state_init=np.array([1, 1, 1]), time_final=5)
    policy_model = PerceptronWithTruncatedNormalNoise(
        dim_input=system.dim_observation,
        dim_output=system.dim_inputs,
        dim_hidden=4,
        n_hidden_layers=1,
        hidden_activation=torch.nn.LeakyReLU(0.2),
        stds=0.1
        * (np.array(system.action_bounds)[:, 1] - np.array(system.action_bounds)[:, 0])
        / 2.0,
        is_truncated_to_output_bounds=True,
        output_activation=regelum.model.MultiplyByConstant(0.5),
        output_bounds=system.action_bounds,
    )
    critic_model = ModelPerceptron(
        dim_input=system.dim_observation,
        dim_output=1,
        dim_hidden=100,
        n_hidden_layers=4,
        hidden_activation=torch.nn.LeakyReLU(0.2),
    )
    running_objective = RunningObjective(
        model=ModelQuadLin("diagonal", weights=np.array([10.0, 10.0, 1.0, 0.0, 0.0]))
    )
    critic_n_epochs: 50
    critic_opt_method = torch.optim.Adam
    critic_opt_method_kwargs = {"lr": 0.001}

    policy_n_epochs: 50
    policy_opt_method = torch.optim.Adam
    policy_opt_method_kwargs = {"lr": 0.1}
    scenario = PPO(
        policy_model=policy_model,
        simulator=simulator,
        sampling_time=0.01,
        running_objective=running_objective,
        critic_n_epochs=1,
        policy_n_epochs=1,
        critic_model=critic_model,
        critic_opt_method_kwargs=critic_opt_method_kwargs,
        policy_opt_method=policy_opt_method,
        policy_opt_method_kwargs=policy_opt_method_kwargs,
        N_episodes=5,
        N_iterations=1,
    )

    scenario.run()

    cfg_scenario = deepcopy(~cfg.scenario)
    del cfg
    cfg_scenario.run()


if __name__ == "__main__":
    JOB_RESULTS = launch()
    print(JOB_RESULTS["result"])
