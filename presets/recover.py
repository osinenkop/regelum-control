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
from rcognita.visualization.vis_3wrobot import (
    Animator3WRobotNI,
    Animator3WRobot,
)
from rcognita.visualization.vis_lunar_lander import (
    AnimatorLunarLander,
)
from rcognita.visualization import plot_multirun
import matplotlib.pyplot as plt
from rcognita.callbacks import ObjectiveCallbackMultirun, TotalObjectiveCallbackMultirun
from rcognita.scenarios import Scenario
import matplotlib.pyplot as plt
import pandas as pd
import dill

np.random.seed(42)


if __name__ == "__main__":
    # plot_multirun.plot_objectives(job_results, PRESET)
    with open("presets/scenario_at_episode_11.dill", "rb") as f:
        scenario = dill.load(f)
        print(scenario.episode_counter)
