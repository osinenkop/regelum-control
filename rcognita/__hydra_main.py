# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import functools
import pickle
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, List, Optional

from omegaconf import DictConfig, open_dict, read_write

from hydra import version
from hydra._internal.deprecation_warning import deprecation_warning
from rcognita.__internal_utils import _run_hydra, get_args_parser
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import _flush_loggers, configure_log
from hydra.types import TaskFunction
import pandas as pd

import os, pickle

_UNSPECIFIED_: Any = object()


def _get_rerun_conf(file_path: str, overrides: List[str]) -> DictConfig:
    msg = "Experimental rerun CLI option, other command line args are ignored."
    warnings.warn(msg, UserWarning)
    file = Path(file_path)
    if not file.exists():
        raise ValueError(f"File {file} does not exist!")

    if len(overrides) > 0:
        msg = "Config overrides are not supported as of now."
        warnings.warn(msg, UserWarning)

    with open(str(file), "rb") as input:
        config = pickle.load(input)  # nosec
    configure_log(config.hydra.job_logging, config.hydra.verbose)
    HydraConfig.instance().set_config(config)
    task_cfg = copy.deepcopy(config)
    with read_write(task_cfg):
        with open_dict(task_cfg):
            del task_cfg["hydra"]
    assert isinstance(task_cfg, DictConfig)
    return task_cfg


def to_dataframe(jobreturns):
    callbacks_ = [res.return_value["callbacks"] for res in jobreturns]
    callbacks = pd.DataFrame(
        callbacks_,
        columns=[callback.__class__.__name__ for callback in callbacks_[0]],
    )
    cfg_ = [res.cfg for res in jobreturns]
    overrides_ = [res.overrides for res in jobreturns]
    result_ = [res.return_value["result"] for res in jobreturns]
    directory_ = [res.return_value["directory"] for res in jobreturns]
    callbacks["cfg"] = cfg_
    callbacks["overrides"] = overrides_
    callbacks["result"] = result_
    callbacks["directory"] = directory_
    return callbacks


def main(
    config_path: Optional[str] = _UNSPECIFIED_,
    config_name: Optional[str] = None,
    version_base: Optional[str] = _UNSPECIFIED_,
) -> Callable[[TaskFunction], Any]:
    """
    :param config_path: The config path, a directory relative to the declaring python file.
                        If config_path is None no directory is added to the Config search path.
    :param config_name: The name of the config (usually the file name without the .yaml extension)
    """

    version.setbase(version_base)

    if config_path is _UNSPECIFIED_:
        if version.base_at_least("1.2"):
            config_path = None
        elif version_base is _UNSPECIFIED_:
            url = "https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path"
            deprecation_warning(
                message=dedent(
                    f"""
                config_path is not specified in @hydra.main().
                See {url} for more information."""
                ),
                stacklevel=2,
            )
            config_path = "."
        else:
            config_path = "."

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args_parser = get_args_parser()
                args = args_parser.parse_args()
                if args.experimental_rerun is not None:
                    cfg = _get_rerun_conf(args.experimental_rerun, args.overrides)
                    task_function(cfg)
                    _flush_loggers()
                else:
                    # no return value from run_hydra() as it may sometime actually run the task_function
                    # multiple times (--multirun)
                    res = _run_hydra(
                        args=args,
                        args_parser=args_parser,
                        task_function=task_function,
                        config_path=config_path,
                        config_name=config_name,
                    )
                    if res:
                        res = to_dataframe(res[0])
                        path = (
                            os.path.abspath(res["directory"][0] + "/..")
                            + "/output.pickle"
                        )
                        with open(path, "wb") as f:
                            pickle.dump(res, f)
                        return res
                    else:
                        return NotImplemented

        return decorated_main

    return main_decorator
