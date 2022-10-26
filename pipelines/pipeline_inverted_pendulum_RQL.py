from pipeline_inverted_pendulum_PG import PipelineInvertedPendulumPG
from rcognita.actors import ActorRQL
from rcognita.critics import CriticActionValue
from rcognita.systems import SysInvertedPendulumPD
from rcognita.models import (
    ModelQuadNoMixTorch,
    ModelNN,
    ModelWeightContainer,
    ModelQuadForm,
)
from rcognita.scenarios import EpisodicScenarioBase
from rcognita.utilities import rc
from rcognita.optimizers import SciPyOptimizer, TorchOptimizer
import numpy as np
from torch import nn
import torch
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from rcognita.predictors import EulerPredictorPendulum


class ModelTorch(ModelNN):
    def __init__(self, dim_observation, dim_action, dim_hidden=32, weights=None):
        super().__init__()

        self.fc1 = nn.Linear(dim_observation + dim_action, dim_hidden, bias=False)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.r3 = nn.ReLU()
        self.fc4 = nn.Linear(dim_hidden, 1, bias=False)

        if weights is not None:
            self.load_state_dict(weights)

        self.double()
        self.cache_weights()

    def forward(self, input_tensor, weights=None):
        if weights is not None:
            self.update(weights)

        x = input_tensor
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        x = self.r3(x)
        x = self.fc4(x)
        x = torch.sum(x ** 2)

        return x


class PipelineInvertedPendulumRQL(PipelineInvertedPendulumPG):
    def initialize_predictor(self):
        self.predictor = EulerPredictorPendulum(
            self.pred_step_size,
            self.system._compute_state_dynamics,
            self.system.out,
            self.dim_output,
            self.prediction_horizon,
        )

    def initialize_system(self):
        self.system = SysInvertedPendulumPD(
            sys_type="diff_eqn",
            dim_state=self.dim_state,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_disturb=self.dim_disturb,
            pars=[self.m, self.g, self.l],
            is_dynamic_controller=self.is_dynamic_controller,
            is_disturb=self.is_disturb,
            pars_disturb=[],
        )
        self.observation_init = self.system.out(self.state_init)

    def initialize_models(self):
        self.actor_model = ModelWeightContainer(weights_init=self.action_init)
        self.critic_model = ModelTorch(self.dim_output, self.dim_input)
        self.running_objective_model = ModelQuadForm(weights=self.R1)

    def initialize_optimizers(self):

        opt_options_torch = {"lr": self.learning_rate_critic}
        opt_options_scipy = {
            "maxiter": 1250,
            "maxfev": 5000,
            "disp": False,
            "adaptive": True,
            "xatol": 1e-7,
            "fatol": 1e-7,
        }

        self.actor_optimizer = SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options_scipy
        )
        self.critic_optimizer = TorchOptimizer(opt_options_torch)

    def initialize_actor_critic(self):
        self.critic = CriticActionValue(
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            data_buffer_size=self.data_buffer_size,
            running_objective=self.running_objective,
            discount_factor=self.discount_factor,
            optimizer=self.critic_optimizer,
            model=self.critic_model,
            sampling_time=self.sampling_time,
        )
        self.actor = ActorRQL(
            self.prediction_horizon,
            self.dim_input,
            self.dim_output,
            self.control_mode,
            self.action_bounds,
            action_init=self.action_init,
            predictor=self.predictor,
            optimizer=self.actor_optimizer,
            critic=self.critic,
            running_objective=self.running_objective,
            model=self.actor_model,
        )

    def initialize_scenario(self):
        self.scenario = EpisodicScenarioBase(
            system=self.system,
            simulator=self.simulator,
            controller=self.controller,
            actor=self.actor,
            critic=self.critic,
            logger=self.logger,
            datafiles=self.datafiles,
            time_final=self.time_final,
            running_objective=self.running_objective,
            no_print=self.no_print,
            is_log=self.is_log,
            is_playback=self.is_playback,
            N_episodes=self.N_episodes,
            N_iterations=self.N_iterations,
            state_init=self.state_init,
            action_init=self.action_init,
        )

    def execute_pipeline(self, **kwargs):
        """
        Full execution routine
        """
        np.random.seed(42)
        super().execute_pipeline(**kwargs)


if __name__ == "__main__":

    PipelineInvertedPendulumRQL().execute_pipeline()
