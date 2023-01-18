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
from rcognita.visualization import plot_multirun
import matplotlib.pyplot as plt
from rcognita.callbacks import HistoricalCallback
from rcognita.scenarios import Scenario
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)


class ObjectiveCallbackMultirun(HistoricalCallback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache = {}
        self.timeline = []
        self.num_launch = 1

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

    @property
    def data(self):
        keys = list(self.cache.keys())
        run_numbers = sorted(list(set([k[0] for k in keys])))
        cache_transformed = {key: list() for key in run_numbers}
        for k, v in self.cache.items():
            cache_transformed[k[0]].append(v)
        return cache_transformed


@r.main(config_name="scenario", callbacks=[ObjectiveCallbackMultirun])
def launch(scenario_config):
    scenario = ~scenario_config
    scenario.run()


if __name__ == "__main__":
    job_results = launch()
    plot_multirun.plot_objectives(job_results, "cartpole")
