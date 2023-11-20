import regelum as rg
import numpy as np
import torch
from regelum.system import InvertedPendulumPD
from regelum.objective import RunningObjective
from regelum.simulator import CasADi
from regelum.scenario import REINFORCE

# Initialize the system and running objective
system = InvertedPendulumPD()
running_objective = RunningObjective(
    model=rg.model.ModelQuadLin("diagonal", weights=[10, 3.0, 0.0])
)

# Instantiate the ReinforceScenario
scenario = REINFORCE(
    policy_model=rg.model.PerceptronWithTruncatedNormalNoise(
        dim_input=system.dim_observation,
        dim_hidden=4,
        n_hidden_layers=2,
        dim_output=system.dim_inputs,
        hidden_activation=torch.nn.LeakyReLU(0.2),
        output_activation=rg.model.MultiplyByConstant(1 / 100),
        output_bounds=system.action_bounds,
        stds=0.1
        * (
            np.array(system.action_bounds)[None, :, 1]
            - np.array(system.action_bounds)[None, :, 0]
        )
        / 2.0,
        is_truncated_to_output_bounds=True,
    ),
    simulator=CasADi(
        system=system,
        state_init=np.array([[3.14, 0]]),
        time_final=10,
        max_step=0.001,
    ),
    policy_opt_method_kwargs={"lr": 0.1},
    sampling_time=0.01,
    running_objective=running_objective,
    N_episodes=4,
    N_iterations=100,
    is_with_baseline=True,
    is_do_not_let_the_past_distract_you=True,
)

# Run the training process
scenario.run()
