import os, sys

# PARENT_DIR = os.path.abspath(__file__ + "/../../")
# sys.path.insert(0, PARENT_DIR)
# CUR_DIR = os.path.abspath(__file__ + "/..")
# sys.path.insert(0, CUR_DIR)


import numpy as np

# os.chdir(PARENT_DIR)
import rcognita as r


# os.chdir(CUR_DIR)

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import dill
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

EXPERIMENT = None
for i, arg in enumerate(sys.argv):
    if "--experiment" in arg:
        EXPERIMENT = arg.split("=")[-1]
        sys.argv.pop(i)


@r.main(config_path="general", config_name="main")
def launch(cfg):
    scenario = ~cfg.scenario
    total_objective = scenario.run()

    if scenario.is_playback:
        # animator = r.vis_3wrobot.Animator3WRobotNI(scenario)
        animator = r.vis_inverted_pendulum.AnimatorInvertedPendulum(scenario)
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
    pass
