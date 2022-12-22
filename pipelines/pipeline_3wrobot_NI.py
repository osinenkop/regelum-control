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

from config_blueprints import Config3WRobotNI
from pipeline_blueprints import PipelineWithDefaults

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
from rcognita.critics import (
    CriticOfActionObservation,
    CriticCALF,
)

from rcognita.utilities import rc


class Pipeline3WRobotNI(PipelineWithDefaults):
    config = Config3WRobotNI

    def initialize_system(self):
        self.system = systems.Sys3WRobotNI(
            sys_type="diff_eqn",
            dim_state=self.dim_state,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_disturb=self.dim_disturb,
            pars=None,
            is_dynamic_controller=self.is_dynamic_controller,
            is_disturb=self.is_disturb,
            pars_disturb=rc.array(
                [
                    [200 * self.sampling_time, 200 * self.sampling_time],
                    [0, 0],
                    [0.3, 0.3],
                ]
            ),
        )

    def initialize_nominal_controller(self):
        self.nominal_controller = controllers.Controller3WRobotNIDisassembledCLF(
            controller_gain=0.5,
            action_bounds=self.action_bounds,
            time_start=self.time_start,
            sampling_time=self.sampling_time,
        )

    def initialize_visualizer(self):

        self.animator = Animator3WRobotNI(
            objects=(
                self.simulator,
                self.system,
                self.nominal_controller,
                self.controller,
                self.datafiles,
                self.logger,
                self.actor_optimizer,
                self.critic_optimizer,
                self.running_objective,
                self.scenario,
            ),
            pars=(
                self.state_init,
                self.action_init,
                self.time_start,
                self.time_final,
                self.state_init,
                self.xMin,
                self.xMax,
                self.yMin,
                self.yMax,
                self.control_mode,
                self.action_manual,
                self.v_min,
                self.omega_min,
                self.v_max,
                self.omega_max,
                self.no_print,
                self.is_log,
                0,
                [],
            ),
        )


if __name__ == "__main__":

    Pipeline3WRobotNI().execute_pipeline()
