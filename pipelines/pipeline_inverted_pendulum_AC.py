import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import matplotlib.pyplot as plt
import numpy as np

import rcognita
import warnings
from datetime import datetime
import pathlib
import csv

from config_blueprints import ConfigInvertedPendulumAC
from pipeline_inverted_pendulum_PG import PipelineInvertedPendulum

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = (
        f"this script is being run using "
        f"rcognita ({rcognita.__version__}) "
        f"located in cloned repository at '{PARENT_DIR}'. "
        f"If you are willing to use your locally installed rcognita, "
        f"run this script ('{os.path.basename(__file__)}') outside "
        f"'rcognita/presets'."
    )
else:
    info = (
        f"this script is being run using "
        f"locally installed rcognita ({rcognita.__version__}). "
        f"Make sure the versions match."
    )
print("INFO:", info)


from rcognita.critics import CriticActionValue
from rcognita.actors import ActorProbabilisticEpisodic

from rcognita.utilities import rc

from rcognita.scenarios import EpisodicScenarioCriticLearn, EpisodicScenario
from rcognita import models, optimizers, loggers, simulator
from copy import deepcopy


class PipelineInvertedPendulumAC(PipelineInvertedPendulum):
    config = ConfigInvertedPendulumAC

    def initialize_logger(self):

        self.datafiles = [None]

        # Do not display annoying warnings when print is on
        if not self.no_print:
            warnings.filterwarnings("ignore")

        self.logger = loggers.loggerInvertedPendulum
        self.logger.N_iterations = self.N_iterations_actor
        self.logger.N_episodes = self.N_episodes_actor
        self.logger_critic = deepcopy(loggers.loggerInvertedPendulum)
        self.logger_critic.N_iterations = self.N_iterations_critic
        self.logger_critic.N_episodes = self.N_episodes_critic

    def initialize_optimizers(self):

        # opt_options_torch = {"lr": 0.00000001}
        opt_options_scipy = {
            "maxiter": 120,
            "maxfev": 5000,
            "disp": False,
            "adaptive": True,
            "xatol": 1e-7,
            "fatol": 1e-7,
        }
        self.actor_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options_scipy
        )
        self.critic_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options_scipy
        )

        # self.critic_optimizer = optimizers.TorchOptimizer(opt_options_torch)

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

        self.actor = ActorProbabilisticEpisodic(
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

    def initialize_models(self):
        self.actor_model = models.ModelGaussianConditional(
            expectation_function=self.safe_controller,
            arg_condition=self.observation_init,
            weights=self.initial_weights,
        )
        # self.critic_model = models.ModelNN(
        #     dim_observation=self.dim_output, dim_action=self.dim_input
        # )
        self.critic_model = models.ModelQuadNoMix(
            self.dim_output + self.dim_input,
            single_weight_max=-1e-3,
            single_weight_min=-1e2,
        )
        self.running_objective_model = models.ModelQuadForm(weights=self.R1)
        self.critic_model.weights = self.critic_model.weight_max
        self.critic_model.update_and_cache_weights(self.critic_model.weights)

    def initialize_scenario(self):
        self.scenario_critic_learn = EpisodicScenarioCriticLearn(
            system=self.system,
            simulator=self.simulator_critic,
            controller=self.controller,
            actor=self.actor,
            critic=self.critic,
            logger=self.logger_critic,
            datafiles=self.datafiles,
            time_final=self.time_final_critic,
            running_objective=self.running_objective,
            no_print=self.no_print,
            is_log=self.is_log,
            is_playback=False,
            N_episodes=self.N_episodes_critic,
            N_iterations=self.N_iterations_critic,
            state_init=self.state_init,
            action_init=self.action_init,
            learning_rate=self.learning_rate_critic,
        )

        self.scenario = EpisodicScenario(
            system=self.system,
            simulator=self.simulator_actor,
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
            N_episodes=self.N_episodes_actor,
            N_iterations=self.N_iterations_actor,
            state_init=self.state_init,
            action_init=self.action_init,
            learning_rate=self.learning_rate_actor,
        )

    def initialize_simulator(self):
        self.simulator_actor = simulator.Simulator(
            sys_type="diff_eqn",
            compute_closed_loop_rhs=self.system.compute_closed_loop_rhs,
            sys_out=self.system.out,
            state_init=self.state_init,
            disturb_init=[],
            action_init=self.action_init,
            time_start=self.time_start,
            time_final=self.time_final,
            sampling_time=self.sampling_time,
            max_step=self.sampling_time / 10,
            first_step=1e-6,
            atol=self.atol,
            rtol=self.rtol,
            is_disturb=self.is_disturb,
            is_dynamic_controller=self.is_dynamic_controller,
        )

        self.simulator_critic = simulator.Simulator(
            sys_type="diff_eqn",
            compute_closed_loop_rhs=self.system.compute_closed_loop_rhs,
            sys_out=self.system.out,
            state_init=self.state_init,
            disturb_init=[],
            action_init=self.action_init,
            time_start=self.time_start,
            time_final=self.time_final_critic,
            sampling_time=self.sampling_time,
            max_step=self.sampling_time / 10,
            first_step=1e-6,
            atol=self.atol,
            rtol=self.rtol,
            is_disturb=self.is_disturb,
            is_dynamic_controller=self.is_dynamic_controller,
        )
        self.simulator = self.simulator_actor

    def execute_pipeline(self, **kwargs):
        self.load_config()
        self.setup_env()
        self.__dict__.update(kwargs)
        self.initialize_system()
        self.initialize_predictor()
        self.initialize_safe_controller()
        self.critic_struct = "NN"
        self.initialize_models()
        self.initialize_objectives()
        self.initialize_optimizers()
        self.initialize_actor_critic()
        self.initialize_controller()
        self.initialize_simulator()
        self.initialize_logger()
        self.initialize_scenario()
        self.scenario_critic_learn.run()
        self.controller.is_fixed_critic_weights = True
        self.scenario.reload_pipeline()
        if not self.no_visual and not self.save_trajectory:
            self.initialize_visualizer()
            self.main_loop_visual()
        else:
            self.scenario.run()
            if self.is_playback:
                self.playback()


if __name__ == "__main__":

    PipelineInvertedPendulumAC().execute_pipeline(
        no_visual=True, is_fixed_critic_weights=False, is_fixed_actor_weights=True,
    )
