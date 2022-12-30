import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
import logging
import numpy as np
import rcognita as r
import omegaconf
from omegaconf import DictConfig, OmegaConf, flag_override
from rcognita.visualization.vis_inverted_pendulum import AnimatorInvertedPendulum
import matplotlib.pyplot as plt


@r.main(config_name="scenario")
def launch(scenario_config):
    scenario = ~scenario_config
    scenario.run()
    if scenario.is_playback:
        animator = AnimatorInvertedPendulum(scenario)
        animator.playback()
        plt.show()


if __name__ == "__main__":
    launch()
