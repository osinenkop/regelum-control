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
from rcognita.visualization.vis_3wrobot import Animator3WRobot
import matplotlib.pyplot as plt
from rcognita.callbacks import Callback
from rcognita.scenarios import Scenario
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)


@r.main(config_name="scenario", callbacks=[ObjectiveCallbackMultirun])
def launch(scenario_config):
    scenario = ~scenario_config
    scenario.run()
    if scenario.repeat_num == 1:
        ObjectiveCallbackMultirun.plot_results()


if __name__ == "__main__":
    launch()
