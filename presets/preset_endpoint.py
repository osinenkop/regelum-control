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
from rcognita.visualization.vis_inverted_pendulum import AnimatorInvertedPendulum
from rcognita.visualization import plot_multirun
import matplotlib.pyplot as plt
from rcognita.callbacks import (
    HistoricalObjectiveCallback,
    TotalObjectiveCallback,
    CriticObjectiveCallback,
    CalfCallback,
    HistoricalObservationCallback,
)
from rcognita.scenarios import Scenario
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import dill
from omegaconf import OmegaConf
from dataclasses import dataclass


np.random.seed(42)

EXPERIMENT = None
for i, arg in enumerate(sys.argv):
    if "--experiment" in arg:
        EXPERIMENT = arg.split("=")[-1]
        sys.argv.pop(i)
        break


@r.main(config_path="general", config_name="main")
def launch(cfg):
    scenario = ~cfg.scenario
    total_objective = scenario.run()

    if scenario.is_playback:
        animator = AnimatorInvertedPendulum(scenario)
        animator.playback()
        plt.show()

    return total_objective


def plot_multirun_total_objective(callbacks, preset_name):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots()
    ax.set_xlabel("episode")
    ax.set_ylabel("Total objective")
    df = pd.DataFrame()
    for callback in callbacks:
        df = pd.concat([df, callback.data], axis=1)

    plt.plot(df)
    plt.grid()
    plt.title(f"{preset_name}")
    plt.show()


if __name__ == "__main__":
    job_results = launch()
