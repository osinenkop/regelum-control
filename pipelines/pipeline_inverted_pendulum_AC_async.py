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

from config_blueprints import ConfigInvertedPendulum
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

from rcognita import (
    controllers,
    animators,
    systems,
    predictors,
    objectives,
    models,
)
from rcognita.loggers import loggerInvertedPendulum
from rcognita.actors import ActorProbabilisticEpisodic, ActorProbabilisticEpisodicAC

from rcognita.critics import CriticActionValue

from rcognita.scenarios import EpisodicScenarioAsyncAC


class PipelineInvertedPendulumAC(PipelineInvertedPendulum):
    config = ConfigInvertedPendulum

    def initialize_logger(self):

        self.datafiles = [None]

        # Do not display annoying warnings when print is on
        if not self.no_print:
            warnings.filterwarnings("ignore")

        self.logger = loggerInvertedPendulum
        self.logger.N_iterations = self.N_iterations
        self.logger.N_episodes = self.N_episodes

    def initialize_system(self):
        self.system = systems.SysInvertedPendulum(
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
        self.observation_init = self.system.out(self.state_init, time=0)

    def initialize_models(self):
        self.actor_model = models.ModelGaussianConditional(
            expectation_function=self.safe_controller,
            arg_condition=self.observation_init,
            weights=self.initial_weights,
        )
        self.critic_model = models.ModelNN(self.dim_output, self.dim_input)
        self.running_objective_model = models.ModelQuadForm(weights=self.R1)

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
        self.actor = ActorProbabilisticEpisodicAC(
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

    def initialize_controller(self):
        self.controller = controllers.RLController(
            time_start=self.time_start,
            sampling_time=self.sampling_time,
            critic_period=self.critic_period,
            actor=self.actor,
            critic=self.critic,
            observation_target=self.observation_target,
            is_fixed_critic_weights=self.is_fixed_critic_weights,
        )

    def initialize_scenario(self):
        self.scenario = EpisodicScenarioAsyncAC(
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
            learning_rate=self.learning_rate,
        )

    def initialize_visualizer(self):
        state_full_init = self.simulator.state_full
        self.scenario.is_playback = False
        self.animator = animators.AnimatorInvertedPendulum(
            objects=(
                self.simulator,
                self.system,
                self.safe_controller,
                self.controller,
                self.datafiles,
                self.scenario,
            ),
            pars=(
                self.state_init,
                self.time_start,
                self.time_final,
                state_full_init,
                self.control_mode,
            ),
        )

        if self.is_playback:
            whole_playback_table = self.scenario.episode_tables
            (iters, episodes, ts, angles, angle_dots, Ms, rs, outcomes) = np.hsplit(
                whole_playback_table[:, :-3], 8
            )
            thetas = whole_playback_table[:, -3:]
            self.animator.set_sim_data(
                iters, episodes, ts, angles, angle_dots, Ms, rs, outcomes, thetas,
            )

    def execute_pipeline(self, **kwargs):
        self.load_config()
        self.setup_env()
        self.__dict__.update(kwargs)
        self.data_buffer_size = int(self.time_final / self.sampling_time)
        self.initialize_system()
        self.initialize_predictor()
        self.initialize_safe_controller()
        self.initialize_models()
        self.initialize_objectives()
        self.initialize_optimizers()
        self.initialize_actor_critic()
        self.initialize_controller()
        self.initialize_simulator()
        self.initialize_logger()
        self.initialize_scenario()
        if not self.no_visual and not self.save_trajectory:
            self.initialize_visualizer()
            self.main_loop_visual()
        else:
            self.scenario.run()
            if self.is_playback:
                self.playback()


if __name__ == "__main__":

    PipelineInvertedPendulumAC().execute_pipeline(
        is_fixed_critic_weights=True
    )  #### it forbids to enter inside a critic update procedure

