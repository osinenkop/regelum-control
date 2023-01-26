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
from rcognita.callbacks import ObjectiveCallback, TotalObjectiveCallback
from rcognita.scenarios import Scenario
import matplotlib.pyplot as plt
import pandas as pd
import dill
import pickle

np.random.seed(42)


def plot_multirun_total_objective(callbacks, preset_name):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total objective")
    df = pd.DataFrame()
    for callback in callbacks:
        df = pd.concat([df, callback.data], axis=1)

    plt.plot(df)
    print(df)
    plt.grid()
    plt.title(f"{preset_name}")
    plt.savefig("outcomes.png", format="png")


if __name__ == "__main__":
    # plot_multirun.plot_objectives(job_results, PRESET)
    with open(
        "/home/gvidon/JoraProgramming/rcognita/presets/callbacks_at_episode_5.dill",
        "rb",
    ) as f:
        callbacks = dill.load(f)
        callbacks = callbacks[1]

        plot_multirun_total_objective([callbacks], "inv_pendulum")
