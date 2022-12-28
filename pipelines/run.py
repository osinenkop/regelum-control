import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import logging
import numpy as np
import rcognita as r
import omegaconf
from omegaconf import DictConfig, OmegaConf, flag_override
from rcognita.visualization.vis_3wrobot import Animator3WRobot
import matplotlib.pyplot as plt


@r.main(
    version_base=None, config_path="../pipelines/3wrobot", config_name="scenario",
)
def launch(scenario):
    scenario_instance = ~scenario
    scenario_instance.run()

    if scenario_instance.is_playback:
        animator = Animator3WRobot(scenario_instance)
        animator.playback()
        plt.show()


if __name__ == "__main__":
    launch()
