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

    # plot_multirun.plot_objectives(job_results, EXPERIMENT)
    with open(job_results["directory"][0] + "/../output.pickle", "rb") as f:
        df = pickle.load(f)

    # observation_naming = (
    #     ~r.ComplementedConfig(df.cfg[0]).observation_naming
    # ).observation
    # observation_history_callback = df.HistoricalObservationCallback[0]
    # observation_history_callback.name_observation_components(observation_naming)
    # observation_history = observation_history_callback.data[0]
    # observation_history.plot(subplots=True, layout=(1, len(observation_naming)))
    # plt.show()
    # total_objective_path = os.path.join(
    #     job_results["directory"][0], "total_objectives.png"
    # )
    # overrides_path = os.path.join(job_results["directory"][0], ".hydra/overrides.yaml")
    # algo = OmegaConf.load(overrides_path)[0].split("=")[1]
    # plot_multirun.plot_objectives(
    #     df.TotalObjectiveCallback,
    #     EXPERIMENT,
    #     os.path.join(
    #         job_results["directory"][0] + f"/../{EXPERIMENT.lower()}_{algo.lower()}.png"
    #     ),
    # )
    # plt.plot(df.TotalObjectiveCallback.values[0].data)
    # plt.grid()
    # plt.xticks(range(1, len(df.TotalObjectiveCallback.values[0].data) + 1))
    # plt.savefig(total_objective_path)

    # callbacks_objective = df.TotalObjectiveCallback
    # plot_multirun_total_objective(callbacks_objective, EXPERIMENT)
    # callbacks_calf = df.CalfCallback
    # for i, callback in enumerate(callbacks_calf):
    #     callback.plot_data()
