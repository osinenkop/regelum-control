"""A flexibly configurable framework for agent-enviroment simulation with a menu of predictive and safe reinforcement learning scenarios.

It is made for researchers and engineers in reinforcement learning and control theory.
"""

__version__ = "0.3.2"

import argparse
import datetime
import glob
import platform
import shelve
import subprocess
import sys
import os
import inspect
import logging
import traceback
import types
import warnings

import matplotlib

matplotlib.use("Agg")

import random
from click import Option
from fsspec import Callback
from matplotlib import interactive
import omegaconf
import rich

from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.resolvers import oc
import re

from typing import Any

from omegaconf import Container
from omegaconf._utils import _DEFAULT_MARKER_
from omegaconf.errors import ConfigKeyError, InterpolationResolutionError


import hashlib

from omegaconf import DictConfig
import rehydra
from rehydra._internal.utils import _locate

import rehydra.core.plugins
import rehydra._internal.config_loader_impl
from scipy.optimize import OptimizeWarning
from rehydra.utils import instantiate as inst


import mlflow

from unittest.mock import Mock


import plotly.graph_objects as go
import json


import tempfile

from pathlib import Path

import numpy

from unittest.mock import MagicMock


from rehydra import main as rehydramain
from . import __internal
from . import callback
from regelum.__internal.base import RegelumBase
from .__internal.metadata import Metadata
from .animation import StateAnimation

from .callback import (
    OnEpisodeDoneCallback,
    OnIterationDoneCallback,
    ConfigDiagramCallback,
    TimeRemainingCallback,
    SaveProgressCallback,
)
import time
import pandas as pd

from .configure import get_user_settings, config_file

# try:
#     import torch
# except (ModuleNotFoundError, ImportError):
#     torch = MagicMock()

import torch

from . import model
from . import data_buffers
from .optimizable import *
from . import critic
from . import scenario
from . import policy
from . import observer
from .event import Event

from rich import print, pretty
from rich.panel import Panel
import rich.logging
from rich.logging import RichHandler

mock = Mock()


ANIMATION_TYPES_SAVE_FORMATS = ["html", "mp4"]
ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK = [
    "playback"
] + ANIMATION_TYPES_SAVE_FORMATS
ANIMATION_TYPES_NONE = [None, "None"]
ANIMATION_TYPES_REQUIRING_ANIMATOR = [
    "live",
] + ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK
ANIMATION_TYPES = ANIMATION_TYPES_NONE + ANIMATION_TYPES_REQUIRING_ANIMATOR


if "REGELUM_DATA_DIR" not in os.environ:
    os.environ["REGELUM_DATA_DIR"] = os.path.abspath(sys.path[0]) + "/regelum_data"
else:
    os.environ["REGELUM_DATA_DIR"] = os.path.abspath(os.environ["REGELUM_DATA_DIR"])

if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = (
        f"file://{os.environ['REGELUM_DATA_DIR']}/mlruns"
    )


def hash_string(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), base=16)


import subprocess, os, platform


def start(filepath):
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", filepath], check=True)
    elif platform.system() == "Windows":  # Windows
        os.startfile(filepath)
    else:  # linux variants
        subprocess.run(["xdg-open", filepath], check=True)


def __memorize_instance(resolver):
    def inner(
        key: str,
        default: Any = _DEFAULT_MARKER_,
        *,
        _parent_: Container,
    ):
        if default == _DEFAULT_MARKER_:
            default = key.strip()
        instance_name = str(default)
        if instance_name in main.objects_created:
            return main.objects_created[instance_name]
        else:
            obj = inst(resolver(key, default=default, _parent_=_parent_), path=default)
            main.objects_created[instance_name] = obj
            return obj

    return inner


def __sub_map(pattern, f, s):
    def map_match(match):
        return f(match.group())

    return re.sub(pattern, map_match, s)


def obtain(obj_repr):
    if not isinstance(obj_repr, str):
        obj_repr = str(obj_repr)
    obj_repr = (
        obj_repr.replace("__QUOTATION__", '"')
        .replace("__APOSTROPHE__", "'")
        .replace("__TILDE__", "~")
    )
    obj_repr = rehydra._internal.core_plugins.file_config_source.numerize_string(
        obj_repr
    )
    pattern = re.compile(r"(\A|[^a-zA-Z0-9\._])[a-zA-Z_][a-zA-Z0-9_]*")
    resolved = []

    def resolve(s):
        if s[0].isalnum() or s[0] == "_":
            prefix = ""
        else:
            prefix = s[0]
            s = s[1:]
        try:
            if s == "np":
                s = "numpy"
            entity = _locate(s)
        except Exception:
            entity = eval(s)
        resolved.append(entity)
        return f"{prefix}resolved[{len(resolved) - 1}]"

    obj_repr_resolved_modules = __sub_map(pattern, resolve, obj_repr)
    return eval(obj_repr_resolved_modules)


OmegaConf.register_new_resolver("same", __memorize_instance(oc.select))
OmegaConf.register_new_resolver(name="get", resolver=obtain)
OmegaConf.register_new_resolver(name="mock", resolver=lambda: mock)


class ComplementedConfig:
    """A config object, generated by ``regelum``'s config scenario."""

    def __init__(
        self, cfg: Union[DictConfig, ListConfig], config_path: str = ""
    ) -> None:
        """Initialize a complemented config.

        Args:
            cfg: rehydra config object
            config_path: path to a config file
        """
        self._rehydra_config = cfg
        self.config_path = config_path
        self._saved_hash = None

    def refresh(self):
        """Reset all cached instances.

        Call this if you made changes to your config dynamically and would like those change to take
        effect in forthcoming instantiations.
        """
        main.objects_created.clear()

    def __str__(self):
        return (
            str(self._rehydra_config)
            .replace("DictConfig", "ComplementedConfigDict")
            .replace("ListConfig", "ComplementedConfigList")
            .replace("${get:", "={")
            .replace("${same:", "~{")
        )

    def __repr__(self):
        return self.__class__.__name__ + "( " + repr(self._rehydra_config) + ")"

    def __hash__(self):
        if not self._saved_hash:
            real_config = omegaconf.OmegaConf.to_container(self._rehydra_config)
            return hash_string(json.dumps(real_config, sort_keys=True))
        else:
            return self._saved_hash

    def copy(self):
        return self.__class__(self._rehydra_config.copy(), config_path=self.config_path)

    def __invert__(self):
        """Instantiate the object described by the config (recursively).

        ``_target_`` has to be specified.
        """
        # if not self.config_path:
        return inst(self._rehydra_config, path=self.config_path)
        # else:
        #     instance_name = self.config_path.strip()
        #     if instance_name in self.objects_created:
        #         return self.objects_created[instance_name]
        #     else:
        #         res = inst(self.__rehydra_config, path=self.config_path)
        #         self.objects_created[instance_name] = res
        #         return res


class ComplementedConfigList(ComplementedConfig):
    """A config object, generated by ``regelum``'s config scenario that corresponds to a complemented version of `ListConfig` from omegaconf."""

    def __contains__(self, item):
        return item in self._rehydra_config

    def __getitem__(self, key):
        cfg = object.__getattribute__(self, "_rehydra_config")
        name = key
        item = cfg[name]
        if isinstance(item, DictConfig) or isinstance(item, ListConfig):
            if self.config_path:
                child_config_path = f"{self.config_path}[{name}]"
            else:
                raise ValueError("Top level list configs are not supported.")
            entity = (
                ComplementedConfigList
                if isinstance(item, ListConfig)
                else ComplementedConfigDict
            )
            item = entity(item, config_path=child_config_path)
        return item

    def __setitem__(self, key, value):
        self._rehydra_config[key] = value

    def __delitem__(self, key):
        del self._rehydra_config[key]


class ComplementedConfigDict(ComplementedConfig):
    """A config object, generated by ``regelum``'s config scenario that corresponds to a complemented version of `DictConfig` from omegaconf."""

    def treemap(self, root="config"):
        def format(
            parent_name, parent_node, parent_node_raw, occupied=None, parent_color=50
        ):
            if occupied is None:
                occupied = set()
            labels = []
            parents = []
            colors = []
            text = []
            if isinstance(parent_node, ComplementedConfigDict):
                items = parent_node.items()
            else:
                items = enumerate(parent_node)
            for name, value in items:
                # check if node as attribute value
                parents.append(parent_name)
                # if type(parent_node_raw) is str:
                #    parent_node_raw = omegaconf.OmegaConf.to_container(parent_node.__rehydra_config)
                if not isinstance(name, int):
                    real_name = (
                        name if name in parent_node_raw else name + "__IGNORE__"
                    ).strip()
                else:
                    real_name = name
                    name = str(name)
                while name in occupied:
                    name += " "
                if isinstance(value, ComplementedConfig):
                    node_raw = parent_node_raw[real_name]
                    if type(node_raw) is str:
                        text.append(node_raw.replace("__IGNORE__", "%%"))
                        node_raw = omegaconf.OmegaConf.to_container(
                            value._rehydra_config
                        )
                    else:
                        text.append("")
                    labels.append(name)
                    colors.append((parent_color * 1.2) % 100)
                    (
                        subnode_parents,
                        subnode_labels,
                        subnode_colors,
                        subnode_text,
                    ) = format(
                        name,
                        value,
                        node_raw,
                        occupied=occupied,
                        parent_color=(parent_color * 1.2) % 100,
                    )
                    for i, subnode_label in enumerate(subnode_labels):
                        if subnode_label in labels or subnode_label in parents:
                            subnode_labels[i] = subnode_labels[i] + " "
                    labels += subnode_labels
                    parents += subnode_parents
                    colors += subnode_colors
                    text += subnode_text
                else:
                    if (
                        type(parent_node_raw[real_name]) is str
                        and "$" in parent_node_raw[real_name]
                    ):
                        text.append(
                            parent_node_raw[real_name].replace("__IGNORE__", "%%")
                        )
                    else:
                        text.append("")
                    colors.append(hash_string(name) % 100)
                    name = f"{name}: {str(value)}"
                    while name in occupied:
                        name += " "
                    labels.append(name)

                occupied.add(labels[-1])
            occupied.add(parent_name)
            return parents, labels, colors, text

        raw_self = omegaconf.OmegaConf.to_container(self._rehydra_config)
        self._saved_hash = hash_string(json.dumps(raw_self, sort_keys=True))
        parents, labels, colors, text = format(
            f"{root} {hex(hash(self))}", self, raw_self
        )
        # parents = [parent[:-1] if "_" in parent else parent for parent in parents]
        # parents = [""] + parents
        # labels = ["config"] + labels
        fig = go.Figure(
            go.Treemap(
                labels=labels,
                parents=parents,
                marker=dict(colors=colors, colorscale="RdBu", cmid=50),
                text=text,
            )
        )
        return fig

    def __contains__(self, item):
        return (
            item in self._rehydra_config or item + "__IGNORE__" in self._rehydra_config
        )

    def __getattr__(self, item):
        cfg = object.__getattribute__(self, "_rehydra_config")
        try:
            name = item
            attr = cfg.__getattr__(name)
        except (IndexError, KeyError, AttributeError, ConfigKeyError):
            name = item + "__IGNORE__"
            attr = cfg.__getattr__(name)
        if isinstance(attr, DictConfig) or isinstance(attr, ListConfig):
            if self.config_path:
                child_config_path = self.config_path + "." + name
            else:
                child_config_path = name
            entity = (
                ComplementedConfigList
                if isinstance(item, ListConfig)
                else ComplementedConfigDict
            )
            attr = entity(attr, config_path=child_config_path)
        return attr

    def __getitem__(self, key):
        cfg = object.__getattribute__(self, "_rehydra_config")
        try:
            name = key
            item = cfg[name]
        except (IndexError, KeyError, AttributeError, ConfigKeyError) as e:
            if isinstance(key, str):
                name = key + "__IGNORE__"
                item = cfg[name]
            else:
                raise e
        if isinstance(item, DictConfig) or isinstance(item, ListConfig):
            if self.config_path:
                child_config_path = self.config_path + "." + name
            else:
                child_config_path = name
            entity = (
                ComplementedConfigList
                if isinstance(item, ListConfig)
                else ComplementedConfigDict
            )
            item = entity(item, config_path=child_config_path)
        return item

    def __setitem__(self, key, value):
        if key + "__IGNORE__" in self._rehydra_config:
            self._rehydra_config[key + "__IGNORE__"] = value
        else:
            self._rehydra_config[key] = value

    def __setattr__(self, key, value):
        if key in ["config_path", "_saved_hash"]:
            object.__setattr__(self, key, value)
            return
        if hasattr(self, "_rehydra_config"):
            if key + "__IGNORE__" in self._rehydra_config:
                self._rehydra_config.__setattr__(key + "__IGNORE__", value)
            else:
                self._rehydra_config.__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    def has_key(self, key):
        return key in self._rehydra_config or key + "__IGNORE__" in self._rehydra_config

    def clear(self):
        self._rehydra_config.clear()

    def update(self, *args, **kwargs):
        for subdict in args:
            for key, value in subdict.values():
                self[key] = value
        for key, value in kwargs.values():
            self[key] = value

    def keys(self):
        return [key.replace("__IGNORE__", "") for key in self._rehydra_config.keys()]

    def items(self):
        return [
            (
                key.replace("__IGNORE__", ""),
                (
                    value
                    if not isinstance(value, DictConfig)
                    and not isinstance(value, ListConfig)
                    else (
                        ComplementedConfigDict
                        if isinstance(value, DictConfig)
                        else ComplementedConfigList
                    )(
                        value,
                        config_path=(
                            key if not self.config_path else f"{self.config_path}.{key}"
                        ),
                    )
                ),
            )
            for key, value in self._rehydra_config.items()
        ]

    def values(self):
        return [
            (
                value
                if not isinstance(value, DictConfig)
                and not isinstance(value, ListConfig)
                else (
                    ComplementedConfigDict
                    if isinstance(value, DictConfig)
                    else ComplementedConfigList
                )(
                    value,
                    config_path=(
                        key if not self.config_path else f"{self.config_path}.{key}"
                    ),
                )
            )
            for key, value in self._rehydra_config.items()
        ]

    def __delitem__(self, key):
        if key + "__IGNORE__" in self._rehydra_config:
            del self._rehydra_config[key + "__IGNORE__"]
        else:
            del self._rehydra_config[key]


class RegelumExitException(Exception):
    """Raised to forcefully shut down current simulation."""

    pass


logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

pretty.install()


logo_ascii = """██████╗ ███████╗ ██████╗ ███████╗██╗     ██╗   ██╗███╗   ███╗
██╔══██╗██╔════╝██╔════╝ ██╔════╝██║     ██║   ██║████╗ ████║
██████╔╝█████╗  ██║  ███╗█████╗  ██║     ██║   ██║██╔████╔██║
██╔══██╗██╔══╝  ██║   ██║██╔══╝  ██║     ██║   ██║██║╚██╔╝██║
██║  ██║███████╗╚██████╔╝███████╗███████╗╚██████╔╝██║ ╚═╝ ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝"""


class main:
    """The decorator used to invoke ``regelum``'s config scenario.

    Use it to decorate your ``main`` function like so:
    ::

         @regelum.main(config_path=..., config_name=...)
         def my_main(config):
             ...

         if __name__ == '__main__':
             my_main()
    """

    callbacks = None
    logger = None
    metadata = None
    config = None
    assignments = []
    weak_assignments = []
    builtin_callbacks = [
        OnEpisodeDoneCallback,
        OnIterationDoneCallback,
        ConfigDiagramCallback,
        TimeRemainingCallback,
        SaveProgressCallback,
    ]
    objects_created = {}

    class RegelumArgumentParser(argparse.ArgumentParser):
        """A parser object designed for handling peculiar nuances of arg parsing while interfacing with rehydra."""

        def __init__(self, *args, **kwargs):
            """Initialize and instance of RegelumArgumentParser.

            :param args: arguments to pass to base
            :param kwargs: keyword arguments to pass to base
            """
            self._stored_args = []
            self._registered_args = []
            self._triggers = {}
            super().__init__(*args, **kwargs)

        def add_argument(self, *args, trigger=None, **kwargs):
            res = super().add_argument(*args, **kwargs)
            self._registered_args = self._registered_args + list(args)
            for arg in args:
                if arg.startswith("--"):
                    key = arg[2:]
                elif arg.startswith("-"):
                    key = arg[1:]
                else:
                    raise ValueError(
                        "Argument name should start with either a '-' or a '--'."
                    )
                key = key.replace("-", "_")
                if key == "help" or key == "h":
                    continue
                self._triggers[key] = trigger if trigger is not None else lambda v: None
            return res

        def parse_args(self):
            if "--single-thread" in sys.argv:
                for i, arg in enumerate(sys.argv):
                    if "--jobs" in arg or "launcher.n_jobs" in arg:
                        print(f"Ingnoring {arg}, since --single-thread was specified.")
                        sys.argv.pop(i)
                        break
            self.old_args = sys.argv
            self._stored_args = [
                arg for arg in sys.argv if arg.split("=")[0] in self._registered_args
            ]
            res = super().parse_args(self._stored_args)
            sys.argv = [arg for arg in sys.argv if arg not in self._stored_args]
            for key in self._triggers:
                self._triggers[key](vars(res)[key])
            self._registered_args = []

            sys.argv.insert(1, "--multirun")
            sys.argv.append("rehydra.job.chdir=True")

            return res

        def restore_args(self):
            sys.argv = self.old_args

    @classmethod
    def post_weak_assignment(cls, key, value):
        cls.weak_assignments.append((key, value))

    @classmethod
    def post_assignment(cls, key, value, weak=False):
        if weak:
            cls.post_weak_assignment(key, value)
        else:
            cls.assignments.append((key, value))

    @classmethod
    def apply_assignments(cls, cfg):
        for key, value in cls.weak_assignments:
            if callable(value):
                value = value(cfg)
            current = eval(f"cfg.{key.replace('%%', '')}")
            if current == "__REPLACE__":
                exec(f"cfg.{key.replace('%%', '')} = value")
        for key, value in cls.assignments:
            if callable(value):
                value = value(cfg)
            exec(f"cfg.{key.replace('%%', '')} = value")

    def __init__(
        self,
        *args,
        logger=None,
        callbacks=None,
        **kwargs,
    ):
        """Create an instance of ``regelum.main``.

        :param config_path: path to the folder containing your config(s)
        :type config_path: str
        :param config_name: the name of your config (without extension)
        :type config_name: str
        :param callbacks: list of uninstantiated callbacks (classes) to be registered
        :type callbacks: list[class], optional
        :param logger: a logger instance to be supplied to callbacks
        :type logger: Logger, optional

        """
        print(
            Panel.fit(
                logo_ascii,
                subtitle=f"v{__version__}",
                title="AIDA Solutions, 2024",
                style="magenta",
            )
        )
        if logger is None:
            logger = logging.getLogger("regelum")
            # logger.info("Try")
        if callbacks is None:
            callbacks = []

        if "--configure" in sys.argv:
            if os.path.exists(config_file):
                os.remove(config_file)
            sys.argv.pop(sys.argv.index("--configure"))

        user_settings = get_user_settings()
        for name in user_settings:
            if name not in os.environ:
                if user_settings[name] is not None:
                    os.environ[name] = str(user_settings[name])

        self.callbacks_ = callbacks + self.builtin_callbacks

        self.parser = self.RegelumArgumentParser(description="Run with config ...")

        self.parser.add_argument("--no-git", action="store_true")

        def disable_logging(flag):
            if flag:
                sys.argv.append("rehydra/job_logging=disabled")

        self.parser.add_argument(
            "--disable-logging", action="store_true", trigger=disable_logging
        )

        def disable_callbacks(flag):
            if flag:
                self.__class__.builtin_callbacks = []
                self.callbacks_ = []

        self.parser.add_argument(
            "--disable-callbacks", action="store_true", trigger=disable_callbacks
        )

        self.parser.add_argument("--enable-streamlit", action="store_true")

        self.parser.add_argument("--interactive", action="store_true")
        self.parser.add_argument("--show-plots", action="store_true")
        self.parser.add_argument("--playback", action="store_true")

        self.parser.add_argument("--fps", default=2.5)

        self.parser.add_argument("--cooldown-factor", default=1.0)

        self.parser.add_argument("--save-animation", action="store_true")
        self.parser.add_argument("--parallel", action="store_true")

        def single_thread(flag):
            if not flag:
                sys.argv.append("rehydra/launcher=joblib")

        self.parser.add_argument(
            "--single-thread", action="store_true", trigger=single_thread
        )

        def sweep(flag):
            if flag:
                sys.argv.append("rehydra/sweeper=ax")
                self.is_sweep = True
            else:
                self.is_sweep = False

        self.parser.add_argument("--sweep", action="store_true")

        def tags(val):
            if val:
                tags_str = val.split(",")
                self.tags = {tag.split(":")[0]: tag.split(":")[1] for tag in tags_str}
            else:
                self.tags = {}

        self.parser.add_argument("--tags", trigger=tags)

        def experiment(val):
            if val:
                self.experiment_name = val

        self.parser.add_argument("--experiment", trigger=experiment)

        def jobs(val):
            if val is not None:
                sys.argv.append(f"rehydra.launcher.n_jobs={val}")

        self.parser.add_argument("--jobs", trigger=jobs)

        self.parser.add_argument("--skip-frames", action="store", default=1, type=int)

        self.tags = {}
        self.experiment_name = "Default"

        self.__class__.is_clear_matplotlib_cache_in_callbacks = True
        for arg in sys.argv:
            if "live" in arg:
                self.__class__.is_clear_matplotlib_cache_in_callbacks = False

        self.args = args
        self.kwargs = kwargs
        self.kwargs["version_base"] = (
            None if "version_base" not in kwargs else kwargs["version_base"]
        )
        self.kwargs["config_path"] = (
            "." if "config_path" not in kwargs else kwargs["config_path"]
        )
        self.kwargs["config_name"] = (
            None if "config_name" not in kwargs else kwargs["config_name"]
        )
        self.__class__.logger = logger

    def __call__(self, old_app):
        def rcognita_main(*args, **kwargs):

            argv = self.parser.parse_args()
            common_dir = tempfile.TemporaryDirectory()
            initial_working_directory = os.getcwd()
            initial_pythonpath = (
                os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""
            )
            script_path = inspect.getsourcefile(old_app)
            no_git = argv.no_git
            skip_frames = argv.skip_frames
            path_main = os.path.abspath(script_path)
            path_parent = "/".join(path_main.split("/")[:-1])
            os.chdir(path_parent)
            path = (
                os.path.abspath(self.kwargs["config_path"])
                if self.kwargs["config_path"] is not None
                else None
            )
            self.kwargs["config_path"] = path

            def app(
                cfg,
                argv=argv,
                logger=self.__class__.logger,
                is_clear_matplotlib_cache_in_callbacks=self.__class__.is_clear_matplotlib_cache_in_callbacks,
            ):
                # args = self.parser.parse_args()
                if "seed" in cfg:
                    seed = cfg["seed"]
                    delattr(cfg, "seed")
                else:
                    seed = 0

                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
                numpy.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                with shelve.open(".report") as f:
                    f["termination"] = "running"
                    f["started"] = datetime.datetime.now()
                    f["args"] = sys.argv
                    f["config_path"] = path
                    f["script_path"] = script_path
                    f["PYTHONPATH"] = initial_pythonpath
                    f["initial_cwd"] = initial_working_directory
                    f["common_dir"] = common_dir.name
                    f["hostname"] = platform.node()
                    f["seed"] = seed
                os.mkdir("gfx")
                os.mkdir(".callbacks")

                with omegaconf.flag_override(cfg, "allow_objects", True):
                    self.__class__.metadata = {
                        "no_git": no_git,
                        "skip_frames": skip_frames,
                        "logger": logger,
                        "script_path": script_path,
                        "config_path": (
                            path + f"/{self.kwargs['config_name']}.yaml"
                            if self.kwargs["config_name"] is not None
                            else None
                        ),
                        "initial_working_directory": initial_working_directory,
                        "initial_pythonpath": initial_pythonpath,
                        "common_dir": common_dir.name,
                        "id": os.getcwd().split("/")[-1],
                        "report": lambda: shelve.open(
                            common_dir.name
                            + "/report_"
                            + os.getcwd().split("/")[-1].zfill(5)
                        ),
                        "pid": os.getpid(),
                        "argv": argv,
                        "main": self,
                    }
                    with Metadata(self.__class__.metadata):
                        self.callbacks = [callback_() for callback_ in self.callbacks_]
                        self.__class__.is_clear_matplotlib_cache_in_callbacks = (
                            is_clear_matplotlib_cache_in_callbacks
                        )
                        self.__class__.logger = logger
                        ccfg = ComplementedConfigDict(cfg)
                        self.apply_assignments(ccfg)
                        if "callbacks" in cfg and not argv.disable_callbacks:
                            for callback_ in cfg.callbacks:
                                callback_ = (
                                    obtain(callback_)
                                    if isinstance(callback_, str)
                                    else callback_
                                )
                                self.callbacks.append(callback_())
                            delattr(cfg, "callbacks")
                        elif "callbacks" in cfg:
                            delattr(cfg, "callbacks")
                        self.__class__.config = ccfg

                        try:
                            for callback_ in self.callbacks:
                                if callback_.cooldown:
                                    callback_.cooldown *= argv.cooldown_factor
                                callback_.on_launch()  # make sure this line is adequate to mlflow functionality
                            with self.__class__.metadata["report"]() as r:
                                r["path"] = os.getcwd()
                                r["pid"] = os.getpid()
                            self.tags.update({"run_path": os.getcwd()})

                            experiment_id = mlflow.set_experiment(
                                self.experiment_name
                            ).experiment_id

                            with mlflow.start_run(
                                experiment_id=experiment_id,
                                tags=self.tags,
                                run_name=" ".join(os.getcwd().split("/")[-3:]),
                            ):
                                overrides = {
                                    line.split("=")[0]
                                    .replace("+", "")
                                    .replace("/", "-"): line.split("=")[1]
                                    for line in OmegaConf.load(
                                        ".rehydra/overrides.yaml"
                                    )
                                }
                                mlflow.log_params(overrides)
                                mlflow.log_artifact("SUMMARY.html")
                                mlflow.log_artifact(".summary")
                                mlflow.log_artifact(".rehydra")
                                res = old_app(ccfg)
                                mlflow.log_artifact(".callbacks")
                                try:
                                    mlflow.log_artifact("__init__.log")
                                except FileNotFoundError:
                                    mlflow.log_artifact(
                                        "conftest.log"
                                    )  ## TO DO: Find a better way to handle this
                                if os.path.exists("callbacks.dill"):
                                    mlflow.log_artifact("callbacks.dill")

                            self.callbacks[0].log("Script terminated successfully.")
                        except RegelumExitException as e:
                            res = e
                        except InterpolationResolutionError as e:
                            with self.__class__.metadata["report"]() as r:
                                r["traceback"] = traceback.format_exc()
                            res = e
                            self.callbacks[0].log(
                                "Script terminated with error. This error occurred when trying to instantiate something from the config."
                            )
                            self.callbacks[0].exception(e)
                        except Exception as e:
                            with self.__class__.metadata["report"]() as r:
                                r["traceback"] = traceback.format_exc()
                            res = e
                            self.callbacks[0].log("Script terminated with error.")
                            self.callbacks[0].exception(e)
                        for callback_ in self.callbacks:
                            try:
                                callback_.on_termination(res)
                            except Exception as e:
                                callback_.log(
                                    f"Termination procedure for {callback_.__class__.__name__} failed."
                                )
                                callback_.exception(e)
                        with shelve.open(".report") as f:
                            f["termination"] = (
                                "successful"
                                if not isinstance(res, Exception)
                                else "".join(traceback.format_tb(res.__traceback__))
                            )
                            f["finished"] = datetime.datetime.now()
                        ccfg.refresh()
                    if argv.sweep:
                        return res
                    else:
                        if argv.playback:
                            filenames = glob.glob(".callbacks/**/*.html")
                            n_animations = len(
                                set([Path(filename).parent for filename in filenames])
                            )
                            latest_files = sorted(filenames, key=os.path.getctime)
                            for filename in latest_files[-n_animations:]:
                                start(filename)
                        if argv.show_plots:
                            start("SUMMARY.html")
                        return {
                            "result": res,
                            "callbacks": self.callbacks,
                            "directory": os.getcwd(),
                        }

            app.__module__ = old_app.__module__
            res = rehydramain(*self.args, **self.kwargs)(app)(*args, **kwargs)
            common_dir.cleanup()
            self.parser.restore_args()
            return res

        return rcognita_main


warnings.filterwarnings("ignore", category=UserWarning, module=rehydra.__name__)
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

array = utils.rg.array
rg = utils.rg


def set_ipython_env(
    callbacks: Optional[List[Callback]] = None, interactive: bool = False
):
    logger = logging.getLogger("regelum")
    logger.setLevel(logging.INFO)
    from dataclasses import dataclass

    @dataclass
    class CallbackContainer:
        callbacks: list

    @dataclass
    class ArgvContainer:
        interactive: list
        parallel: bool

    RegelumBase._metadata = {
        "logger": logger,
        "argv": ArgvContainer(interactive, False),
    }
    callbacks = [callback() for callback in callbacks]
    RegelumBase._metadata = {
        "logger": logger,
        "main": CallbackContainer(callbacks),
        "argv": ArgvContainer(interactive, False),
    }
    regelum.main.is_clear_matplotlib_cache_in_callbacks = True
    return callbacks
