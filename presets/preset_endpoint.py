import os, sys

<<<<<<< HEAD
PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
=======
#PARENT_DIR = os.path.abspath(__file__ + "/../../")
#sys.path.insert(0, PARENT_DIR)
#CUR_DIR = os.path.abspath(__file__ + "/..")
#sys.path.insert(0, CUR_DIR)


>>>>>>> 086c847a82de2fe103228ffdd9de4e4f839826b1
import numpy as np
#os.chdir(PARENT_DIR)
import rcognita as r
<<<<<<< HEAD
from rcognita.visualization.vis_3wrobot import (
    Animator3WRobotNI,
    Animator3WRobot,
)
from rcognita.visualization.vis_lunar_lander import (
    AnimatorLunarLander,
)
from rcognita.visualization.vis_inverted_pendulum import AnimatorInvertedPendulum
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
=======


#os.chdir(CUR_DIR)

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import dill

>>>>>>> 086c847a82de2fe103228ffdd9de4e4f839826b1


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
        animator = r.visualization.vis_inverted_pendulum.AnimatorInvertedPendulum(scenario)
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
    print("whatever")
