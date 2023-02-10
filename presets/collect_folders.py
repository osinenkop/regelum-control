import os, sys

PARENT_DIR = os.path.abspath(os.getcwd() + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(os.getcwd() + "/..")
sys.path.insert(0, CUR_DIR)

import datetime

# import rcognita.scripts as scripts
import omegaconf
from pathlib import Path

import json
import pandas as pd
from collections import defaultdict
import socket
import numpy as np
import os
from dateutil.parser import parse
from collections import ChainMap, defaultdict


def parse_datetime(run_path):
    return (
        str(run_path).split("/")[-3] + "T" + str(run_path).split("/")[-2].split("_")[0]
    )


def get_run_info(path_run):
    try:
        overrides_path = Path(path_run) / ".hydra" / "overrides.yaml"
        if os.path.isfile(overrides_path):
            return dict(
                [
                    override.split("=")
                    for override in omegaconf.OmegaConf.load(overrides_path)
                ]
            )
    except:
        return dict()


def group_runs(run_paths):
    if len(run_paths) == 0:
        return None, None
    run_infos = {run_path: get_run_info(run_path) for run_path in sorted(run_paths)}
    groups = defaultdict(lambda: defaultdict(list))

    for run_path, info in run_infos.items():
        try:
            groups[info["controller"]][info["system"]].append(run_path)
        except:
            continue
    return groups, run_infos


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(str(string), fuzzy=fuzzy)
        return True

    except ValueError:
        return False


systems = [
    "2tank",
    "3wrobot_ni",
    "3wrobot",
    "cartpole",
    "inv_pendulum",
    "kin_point",
    "lunar_lander",
]

namings = {
    system: eval(
        omegaconf.OmegaConf.load(
            "general/observation_naming/naming_" + system + ".yaml"
        )["observation"][2:]
    )
    for system in systems
}

time_finals = {
    system: omegaconf.OmegaConf.load(f"general/system_specific/spec_{system}.yaml")[
        "time_final"
    ]
    for system in systems
}

runs = []
for folder_day in list(Path("multirun").iterdir()):
    if is_date(folder_day, fuzzy=True):
        for multirun_path in folder_day.iterdir():
            for run_path in multirun_path.iterdir():
                if os.path.isdir(run_path):
                    runs.append(run_path.resolve())
groupped_runs, run_infos = group_runs(runs)
data = list()

for controller in groupped_runs:
    for system in groupped_runs[controller]:
        for run_path in groupped_runs[controller][system]:
            total_objectives_path = (
                Path(run_path)
                / ".callbacks"
                / "TotalObjectiveCallback"
                / "Total_Objective.h5"
            )
            if os.path.isfile(total_objectives_path):
                total_objectives = pd.read_hdf(total_objectives_path)
                finished_episodes = total_objectives.episode.max()
                best_episode = total_objectives.set_index("episode")[
                    "objective"
                ].idxmin()
                best_total_objective = total_objectives["objective"].min()
                best_observations_path = (
                    Path(run_path)
                    / ".callbacks"
                    / "HistoricalObservationCallback"
                    / f"observations_in_episode_{str(best_episode).zfill(5)}.h5"
                )
                if os.path.isfile(best_observations_path):
                    best_observations = pd.read_hdf(best_observations_path)
                    valid_columns = (
                        list(best_observations.columns[3:]) == namings[system]
                    )
                    time_final = best_observations.time.max()
                    valid_time_final = np.isclose(
                        time_final, time_finals[system], atol=1e-1
                    )
                else:
                    best_observations = None
                    valid_columns = None
                    time_final = None

            else:
                total_objectives = None
                finished_episodes = None
                best_episode = None
                best_total_objective = None
                best_observations = None
                valid_columns = None
                time_final = None
                valid_time_final = None

            item = {
                "hostname": socket.gethostname(),
                "controller": controller,
                "system": system,
                "seed": run_infos[run_path].get("+seed"),
                "timestamp": datetime.datetime.strptime(
                    parse_datetime(run_path), "%Y-%m-%dT%H-%M-%S"
                ),
                "run_path": run_path,
                "finished_episodes": finished_episodes,
                "time_final": time_final,
                "best_episode": best_episode,
                "best_total_objective": best_total_objective,
                "valid_columns": valid_columns,
                "valid_time_final": valid_time_final,
                "overrides": json.dumps(run_infos[run_path]),
                "total_objectives": total_objectives,
                "best_observations": best_observations,
            }

            data.append(item)

df = pd.DataFrame.from_records(data)
df.to_hdf(f"{socket.gethostname()}.h5", key="data")
