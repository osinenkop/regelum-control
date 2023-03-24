import os, sys
import mlflow

# PARENT_DIR = os.path.abspath(__file__ + "/../../")
# sys.path.insert(0, PARENT_DIR)
# CUR_DIR = os.path.abspath(__file__ + "/..")
# sys.path.insert(0, CUR_DIR)


import numpy as np

# os.chdir(PARENT_DIR)
import rcognita as rc
import time

# os.chdir(CUR_DIR)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import pandas as pd
import pickle
import dill
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


@rc.main(config_path="general", config_name="main")
def launch(cfg):
    scenario = ~cfg.scenario
    total_objective = scenario.run()

    if scenario.is_playback:
        animator = rc.vis_3wrobot.Animator3WRobot(scenario, save_format="mp4", fps=10)
        animator.playback()
        # plt.rcParams["animation.frame_format"] = "svg"
        # with open(
        #     "/mnt/abolychev/rcognita-research-calf/rcognita-safe-controllers/presets/anim.html",
        #     "w",
        # ) as f:
        #     f.write(
        #         f"<html><head><title>{r.vis_3wrobot.Animator3WRobot.__class__.__name__}</title></head><body>{animator.anm.to_jshtml()}</body></html>"
        #     )

        # mlflow.log_artifact(
        #     "/mnt/abolychev/rcognita-research-calf/rcognita-safe-controllers/presets/anim.html"
        # )
        # writer = FFMpegWriter(
        #     fps=10, codec="libx264", extra_args=["-crf", "27", "-preset", "ultrafast"]
        # )
        # start = time.time()
        # animator.anm.save(
        #     "/mnt/abolychev/rcognita-research-calf/rcognita-safe-controllers/presets/anim.mp4",
        #     writer=writer,
        #     dpi=100,
        # )
        # print(time.time() - start)
        # mlflow.log_artifact(
        #     "/mnt/abolychev/rcognita-research-calf/rcognita-safe-controllers/presets/anim.mp4"
        # )

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
