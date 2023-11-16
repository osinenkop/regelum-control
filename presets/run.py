import regelum as rg
import numpy as np

from regelum.system import InvertedPendulumPD
import torch
from regelum.objective import RunningObjective
from regelum.simulator import CasADi


@rg(config_path="stable-presets", config_name="main")
def launch(cfg):
    system = InvertedPendulumPD()
    pipeline = rg.pipeline.PPOPipeline(
        policy_model=rg.model.PerceptronWithTruncatedNormalNoise(
            dim_input=system.dim_observation,
            dim_hidden=4,
            n_hidden_layers=2,
            dim_output=system.dim_inputs,
            hidden_activation=torch.nn.LeakyReLU(0.2),
            output_activation=rg.model.MultiplyByConstant(1 / 100),
            output_bounds=system.action_bounds,
            stds=[[2.5]],
            is_truncated_to_output_bounds=True,
        ),
        critic_model=rg.model.ModelPerceptron(
            dim_input=system.dim_observation,
            dim_hidden=100,
            n_hidden_layers=4,
            dim_output=1,
        ),
        simulator=CasADi(
            system=system,
            state_init=np.array([[3.14, 0]]),
            time_final=10,
            max_step=0.001,
        ),
        critic_n_epochs=50,
        policy_n_epochs=50,
        critic_opt_method_kwargs={"lr": 0.0001},
        policy_opt_method_kwargs={"lr": 0.01},
        sampling_time=0.01,
        running_objective=RunningObjective(
            model=rg.model.ModelQuadLin("diagonal", weights=[10, 3.0, 0.0])
        ),
    )

    pipeline.run()


if __name__ == "__main__":
    job_results = launch()
    pass
