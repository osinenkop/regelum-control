import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

from rcognita.visualization.vis_3wrobot import Animator3WRobot
import pathlib
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
import rcognita

from config_blueprints import Config3WRobot
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
)
from datetime import datetime
from rcognita.utilities import on_key_press, rc


class Pipeline3WRobot(PipelineWithDefaults):
    config = Config3WRobot

    def initialize_system(self):
        self.system = systems.Sys3WRobot(
            sys_type="diff_eqn",
            dim_state=self.dim_state,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_disturb=self.dim_disturb,
            pars=[self.m, self.I],
            is_dynamic_controller=self.is_dynamic_controller,
            is_disturb=self.is_disturb,
            pars_disturb=None,
        )

    def initialize_nominal_controller(self):
        if self.nominal_controller_type == "CLF":
            self.nominal_controller = controllers.Controller3WRobotDisassembledCLF(
                self.m,
                self.I,
                controller_gain=0.5,
                action_bounds=self.action_bounds,
                time_start=self.time_start,
                sampling_time=self.sampling_time,
                max_iters=100,
            )
        elif self.nominal_controller_type == "PID":
            self.nominal_controller = controllers.Controller3WRobotPID(
                state_init=self.state_init,
                params=[self.m, self.I],
                action_bounds=self.action_bounds,
                sampling_time=self.sampling_time,
            )


    def initialize_visualizer(self):

        self.animator = Animator3WRobot(
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
                self.F_min,
                self.M_min,
                self.F_max,
                self.M_max,
                self.no_print,
                self.is_log,
                0,
                [],
            ),
        )


def main():
    Pipeline3WRobot().execute_pipeline()


if __name__ == "__main__":
    main()
