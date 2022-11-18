import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

from rcognita.visualization import animator
import pathlib
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
import rcognita

from config_blueprints import Config3WRobotNI, Config3WRobot
from pipeline_3wrobot_NI import Pipeline3WRobotNI
from pipeline_3wrobot import Pipeline3WRobot

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
    simulator,
    systems,
    loggers,
    predictors,
    optimizers,
    objectives,
    models,
    utilities,
)
from rcognita.loggers import logger3WRobotNI
from datetime import datetime
from rcognita.utilities import on_key_press
from rcognita.actors import (
    ActorCALF,
    ActorMPC,
    ActorRQL,
    ActorSQL,
)
from rcognita.visualization.vis_3wrobot import Animator3WRobotNI
from rcognita.critics import CriticOfActionObservation, CriticCALF, CriticOfObservation

from rcognita.utilities import rc
from copy import deepcopy


class Pipeline3WRobotNICALF(Pipeline3WRobotNI):
    config = Config3WRobotNI

    def initialize_nominal_controller(self):
        self.nominal_controller = controllers.Controller3WRobotNIMotionPrimitive(K=3)

    def initialize_actor_critic(self):
        self.critic = CriticCALF(
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            data_buffer_size=self.data_buffer_size,
            running_objective=self.running_objective,
            discount_factor=self.discount_factor,
            optimizer=self.critic_optimizer,
            model=self.critic_model,
            predictor=self.predictor,
            observation_init=self.state_init,
            safe_controller=self.nominal_controller,
            penalty_param=self.penalty_param,
            sampling_time=self.sampling_time,
            critic_regularization_param=self.critic_regularization_param,
        )
        self.actor = ActorCALF(
            self.nominal_controller,
            self.prediction_horizon,
            self.dim_input,
            self.dim_output,
            self.control_mode,
            action_bounds=self.action_bounds,
            action_init=self.action_init,
            predictor=self.predictor,
            optimizer=self.actor_optimizer,
            critic=self.critic,
            running_objective=self.running_objective,
            model=self.actor_model,
            penalty_param=self.penalty_param,
            actor_regularization_param=self.actor_regularization_param,
        )

    def initialize_optimizers(self):
        opt_options = {
            "maxiter": 150,
            "maxfev": 5000,
            "disp": False,
            "adaptive": True,
            "xatol": 1e-7,
            "fatol": 1e-7,
        }
        self.actor_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options
        )
        self.critic_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options,
        )

    def initialize_controller(self):
        if self.control_mode == "nominal":
            self.controller = self.nominal_controller
        else:

            self.controller = controllers.CALFController(
                time_start=self.time_start,
                sampling_time=self.sampling_time,
                critic_period=self.critic_period,
                actor=self.actor,
                critic=self.critic,
                observation_target=None,
            )


if __name__ == "__main__":
    pipeline = Pipeline3WRobotNICALF()
    pipeline.execute_pipeline()

