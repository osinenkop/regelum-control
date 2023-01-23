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
from rcognita.visualization import plot_multirun
import matplotlib.pyplot as plt
from rcognita.callbacks import ObjectiveCallbackMultirun, TotalObjectiveCallbackMultirun
from rcognita.scenarios import Scenario
import matplotlib.pyplot as plt
import pandas as pd
import pickle

np.random.seed(42)

PRESET = None
for i, arg in enumerate(sys.argv):
    if "--preset" in arg:
        PRESET = arg.split("=")[-1]
        sys.argv.pop(i)
        break


@r.main(
    config_path=f"./{PRESET}",
    config_name=f"scenario",
    callbacks=[ObjectiveCallbackMultirun, TotalObjectiveCallbackMultirun],
)
def launch(scenario_config):
    scenario = ~scenario_config
    outcome = scenario.run()
    return outcome
    # if scenario.is_playback:
    #     animator = AnimatorLunarLander(scenario)
    #     animator.playback()
    #     plt.show()


if __name__ == "__main__":
    job_results = launch()
    # plot_multirun.plot_objectives(job_results, PRESET)
    try:
        with open(job_results["directory"][0] + "/../output.pickle", "rb") as f:
            df = pickle.load(f)

        callbacks = df.TotalObjectiveCallbackMultirun
        plt.plot(callbacks[0].data)
        plt.show()
    except:
        pass
