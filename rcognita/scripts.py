import distutils.errors
import os
import datetime
from distutils.dir_util import copy_tree, remove_tree
import omegaconf
from pathlib import Path
from collections import ChainMap, defaultdict
import socket
import shelve

def aggregate_multiruns(from_=None, till=None, recent=None, multirun_path="multirun"):
    assert recent or from_, "Please specify runs to aggregate"
    assert not recent or not from_
    initial_cwd = os.getcwd()
    os.chdir(multirun_path)
    by_date = sorted(os.listdir("."))
    runs = []
    if isinstance(recent, datetime.timedelta):
        till = datetime.datetime.now()
        from_ = till - recent
        recent = None
    if from_:
        if not till:
            till = datetime.datetime.now()
        from_idx = None
        while not from_idx:
            try:
                from_idx = by_date.index(str(from_.date()))
            except:
                from_idx = None
                from_ += datetime.timedelta(days=1)
            if from_ > till:
                os.chdir(initial_cwd)
                return []
        till_idx = None
        while not till_idx:
            try:
                till_idx = by_date.index(str(till.date()))
            except:
                till_idx = None
                till -= datetime.timedelta(days=1)
            if from_ > till:
                os.chdir(initial_cwd)
                return []
        by_date = by_date[from_idx : till_idx + 1]
        runs = []
        for date_dir in by_date:
            os.chdir(date_dir)
            appended_runs = list(
                map(lambda s: date_dir + "/" + s, sorted(os.listdir(".")))
            )
            for run in appended_runs:
                if (
                    from_
                    <= datetime.datetime.strptime(run.split("_")[0], "%Y-%m-%d/%H-%M-%S")
                    <= till
                ):
                    runs.append(run)
            os.chdir("..")
    elif recent:
        while recent != 0:
            last = by_date.pop(-1)
            os.chdir(last)
            appended_runs = sorted(os.listdir("."))[-recent:]
            recent -= len(appended_runs)
            runs += list(map(lambda s: last + "/" + s, appended_runs))
            os.chdir("..")
    os.chdir(initial_cwd)
    return runs


def merge_runs(
    from_=None,
    till=None,
    recent=None,
    name=None,
    symlink=False,
    multirun_path="multirun",
):
    runs = aggregate_multiruns(
        from_=from_, till=till, recent=recent, multirun_path=multirun_path
    )
    if name is None:
        if from_:
            if not till:
                till = datetime.datetime.now()
            name = f"merged runs {str(from_).split('.')[0]}-{str(till).split('.')[0]}"
        elif recent:
            name = f"merged runs {str(datetime.datetime.now()).split('.')[0]} recent {recent}"
    os.mkdir(name)
    path = os.path.abspath(name)
    if not runs:
        return
    os.chdir("multirun")
    subrun_ids = list(filter(lambda x: x.isnumeric(), os.listdir(runs[0])))
    subrun_paths = {}
    for subrun_id in subrun_ids:
        subrun_paths[subrun_id] = path + "/" + subrun_id
        os.mkdir(subrun_paths[subrun_id])
    for run in runs:
        os.chdir(run)
        for subrun_id in subrun_ids:
            try:
                if symlink:
                    src = os.path.abspath(subrun_id)
                    if not os.path.isdir(src):
                        raise RuntimeError()
                    os.symlink(
                        src, subrun_paths[subrun_id] + "/" + run.replace("/", "_")
                    )
                else:
                    copy_tree(
                        subrun_id, subrun_paths[subrun_id] + "/" + run.replace("/", "_")
                    )
            except (distutils.errors.DistutilsError, RuntimeError) as e:
                os.chdir("../../..")
                remove_tree(path)
                raise RuntimeError(
                    f"Failed to merge runs. Run {run} is not compatible with {runs[0]}. Perhaps select "
                    f"runs are not a part of the same experiment."
                )
        os.chdir("../..")
    os.chdir("..")


def copy_runs(
    output_folder, multirun_folder="multirun", from_=None, till=None, recent=None
):
    groupped_multiruns, run_infos = group_runs(
        multirun_folder, from_=from_, till=till, recent=recent
    )

    if run_infos is None:
        raise ValueError("Can't find runs")

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for controller in groupped_multiruns:
        (Path(output_folder) / controller).mkdir(parents=True, exist_ok=True)
        for system in groupped_multiruns[controller]:
            for orig_run_path in groupped_multiruns[controller][system]:
                copy_to_folder = Path(output_folder) / controller / system
                copy_to_folder.mkdir(parents=True, exist_ok=True)

                destination_run_path = get_copy_to_path_of_run(
                    orig_run_path,
                    run_infos[orig_run_path],
                    copy_to_folder,
                )

                copy_tree(str(orig_run_path), str(destination_run_path))


def get_copy_to_path_of_run(original_run_path, overrides, copy_to_folder):
    paths = os.listdir(Path(copy_to_folder).resolve())
    if len(paths) == 0:
        return Path(copy_to_folder) / construnct_run_foldername(
            original_run_path, overrides, idx=0
        )

    basis_foldername = construnct_run_foldername(original_run_path, overrides)

    max_idx = -1
    for p in paths:
        if basis_foldername in p:
            max_idx = max(int(p.split("_")[-1]), max_idx)

    return Path(copy_to_folder) / construnct_run_foldername(
        original_run_path, overrides, idx=max_idx + 1
    )


def group_runs(multirun_folder="multirun", from_=None, till=None, recent=None):
    multirun_paths = aggregate_multiruns(
        from_=from_, till=till, recent=recent, multirun_path=multirun_folder
    )
    if len(multirun_paths) == 0:
        return None, None

    run_infos = dict(
        ChainMap(
            *[
                get_multirun_info(Path(multirun_folder).resolve() / multirun_path)
                for multirun_path in multirun_paths
            ]
        )
    )
    groups = defaultdict(lambda: defaultdict(list))

    for run_path, info in run_infos.items():
        if info.get("controller") is None or info.get("system") is None:
            continue
            
        groups[info["controller"]][info["system"]].append(run_path)

    return groups, run_infos

def get_status(run_path):
    if not os.path.isfile(Path(run_path) / ".report"):
        return "undef"
    
    with shelve.open(str(Path(run_path) / '.report')) as f:
        status = f["termination"] 

    if status == "successful":
        return "success"
    
    if status == "running":
        return "running"
    
    return "error"
    
def construnct_run_foldername(run_path, overrides, idx=None):
    return "_".join(
        [
            parse_datetime(run_path),
            get_status(run_path),
            get_seed(overrides),
            socket.gethostname(),
        ]
    ) + ("" if idx is None else "_" + str(idx).zfill(3))


def get_seed(overrides: dict):
    return "seed_" + (
        "N" if overrides.get("+seed") is None else str(overrides.get("+seed"))
    )


def parse_datetime(run_path):
    return run_path.split("/")[-3] + "T" + run_path.split("/")[-2].split("_")[0]


def get_run_info(path_run):
    overrides_path = Path(path_run) / ".hydra" / "overrides.yaml"
    if os.path.isfile(overrides_path):
        return dict(
            [
                override.split("=")
                for override in omegaconf.OmegaConf.load(overrides_path)
            ]
        )


def get_multirun_info(path_multirun):
    paths = list(Path(path_multirun).iterdir())
    if len(paths) > 0:
        return {
            str(run_path.resolve()): get_run_info(run_path)
            for run_path in sorted(paths)
            if get_run_info(run_path) is not None
        }


def get_folder_mapping():
    """
    Valid only if called from 'presets' folder"
    """
    try:
        config_append_to_path = "/.hydra/overrides.yaml"
        folder_to_parse = [x for x in os.listdir() if "merged" in x]
        path_to_parse = os.path.abspath(folder_to_parse[0] + "/0/")
        os.chdir(path_to_parse)
        experiment_folders = os.listdir(path_to_parse)
        folder_mapping = dict()
        for experiment in experiment_folders:
            config_path = experiment + config_append_to_path
            config = omegaconf.OmegaConf.load(config_path)
            controller = config[0].split("=")[1]
            system = config[1].split("=")[1]
            folder_mapping[experiment] = f"{controller}-{system}"
        os.chdir("../../")
        return folder_mapping

    except Exception as e:
        print(e)
        os.chdir("../../")
        return 0
