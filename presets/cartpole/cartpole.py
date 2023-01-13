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


class ObjectiveCallbackMultirun(Callback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    cache = {}
    timeline = []
    num_launch = 1

    def perform(self, obj, method, output):
        if isinstance(obj, Scenario) and method == "perform_post_step_operations":
            self.log(f"Current objective: {output[0]}, observation: {output[1]}")
            key = (self.num_launch, obj.time)
            if key in self.cache.keys():
                self.num_launch += 1
                key = (self.num_launch, obj.time)

            self.cache[key] = output[0]
            if self.timeline != []:
                if self.timeline[-1] < key[1]:
                    self.timeline.append(key[1])

            else:
                self.timeline.append(key[1])

    @classmethod
    def cache_transform(cls):
        keys = list(cls.cache.keys())
        run_numbers = sorted(list(set([k[0] for k in keys])))
        cache_transformed = {key: list() for key in run_numbers}
        for k, v in cls.cache.items():
            cache_transformed[k[0]].append(v)
        return cache_transformed

    @classmethod
    def plot_results(cls):
        df = pd.DataFrame(cls.cache_transform())
        df["time"] = cls.timeline
        df.set_index("time", inplace=True)
        print(df)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.subplots()
        ax.set_xlabel("time")
        ax.set_ylabel("running cost")
        plt.title("MPC")

        ci = 95

        low = np.percentile(df.values, 50 - ci / 2, axis=1)
        high = np.percentile(df.values, 50 + ci / 2, axis=1)

        plt.fill_between(df.index, low, high, color="r", alpha=0.2)

        ax.plot(df.index, df.values, color="r", alpha=0.2)
        df["mean_traj"] = df.mean(axis=1)
        ax.plot(df.index, df.mean_traj.values, color="b", label="mean running cost")
        plt.legend()
        plt.show()


@r.main(config_name="scenario", callbacks=[ObjectiveCallbackMultirun]) 
def launch(scenario_config):
    scenario = ~scenario_config
    scenario.run()
    if scenario.repeat_num == 49:
        ObjectiveCallbackMultirun.plot_results()


if __name__ == "__main__":
    launch()
