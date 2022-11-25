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

    def initialize_optimizers(self):
        # opt_options = {
        #     "maxiter": 150,
        #     "maxfev": 5000,
        #     "disp": False,
        #     "adaptive": True,
        #     "xatol": 1e-7,
        #     "fatol": 1e-7,
        # }
        # self.actor_optimizer = optimizers.SciPyOptimizer(
        #     opt_method="SLSQP", opt_options=opt_options
        # )
        # self.critic_optimizer = optimizers.SciPyOptimizer(
        #     opt_method="SLSQP", opt_options=opt_options,
        # )

        opt_options = {
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.print_level": 0,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-2,
        }

        self.actor_optimizer = optimizers.CasADiOptimizer(
            opt_method="ipopt", opt_options=opt_options
        )
        self.critic_optimizer = optimizers.CasADiOptimizer(
            opt_method="ipopt", opt_options=opt_options,
        )


if __name__ == "__main__":
    pipeline = Pipeline3WRobotNICALF()
    pipeline.execute_pipeline()

