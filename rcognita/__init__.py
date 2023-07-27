"""A flexibly configurable framework for agent-enviroment simulation with a menu of predictive and safe reinforcement learning controllers.

It is made for researchers and engineers in reinforcement learning and control theory.
"""

__version__ = "0.2.1"

import argparse
import datetime
import platform
import shelve
import sys
import os
import inspect
import logging
import traceback
import types
import warnings

import random
import omegaconf

from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.resolvers import oc
import re

from typing import Any

from omegaconf import Container
from omegaconf._utils import _DEFAULT_MARKER_
from omegaconf.errors import ConfigKeyError, InterpolationResolutionError

# from omegaconf.grammar_parser import *

from recursive_monkey_patch import monkey_patch

import hashlib

from . import __utilities



#from . import controllers
#from . import systems
#from . import simulator
#from . import systems
#from .visualization import animator
#from . import __utilities
#from . import models
#from . import predictors
#from . import policies
#from . import visualization
#from . import optimizable
#from . import scenarios
#from . import critics
#from . import objectives
#from .optimizable import *
#from .optimizable.core import *
#from .visualization import *
import mlflow
from unittest.mock import Mock, MagicMock
from hydra._internal.utils import _locate

import hydra.core.plugins
import hydra._internal.config_loader_impl
from scipy.optimize import OptimizeWarning
from . import _Plugins__fake_file_config_source
from . import __fake_plugins
from . import __fake_config_loader_impl
from .__gui_server import __file__ as gui_script_file

from . import __instantiate as inst


import plotly.graph_objects as go
import json

import tempfile

from multiprocessing import Process
import numpy




import mlflow
from unittest.mock import Mock
from hydra._internal.utils import _locate

from . import __instantiate as inst

import plotly.graph_objects as go
import json


import tempfile

from multiprocessing import Process
import numpy

from unittest.mock import MagicMock

# main = MagicMock()

#from . import optimizable
#from . import scenarios
#from . import critics
#from . import objectives
#from .optimizable import *
#from .optimizable.core import *
#from .visualization import *
#from . import policies

from .__hydra_main import main as hydramain

from .callbacks import (
    OnEpisodeDoneCallback,
    OnIterationDoneCallback,
    TimeCallback,
    ConfigDiagramCallback,
    TimeRemainingCallback,
    SaveProgressCallback,
)
import time
import pandas as pd

try:
    import torch
except (ModuleNotFoundError, ImportError):
    torch = MagicMock()


from . import models
from . import data_buffers
from .optimizable import *
from . import critics

mock = Mock()

monkey_patch(__fake_plugins, hydra.core.plugins)
monkey_patch(__fake_config_loader_impl, hydra._internal.config_loader_impl)

ANIMATION_TYPES_SAVE_FORMATS = ["html", "mp4"]
ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK = [
    "playback"
] + ANIMATION_TYPES_SAVE_FORMATS
ANIMATION_TYPES_NONE = [None, "None"]
ANIMATION_TYPES_REQUIRING_ANIMATOR = [
    "live",
] + ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK
ANIMATION_TYPES = ANIMATION_TYPES_NONE + ANIMATION_TYPES_REQUIRING_ANIMATOR


def hash_string(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), base=16)


"""TODO:
NEED COMMENT SECTION
DOES THIS METHOD DO?
"""


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
            obj = inst.instantiate(
                resolver(key, default=default, _parent_=_parent_), path=default
            )
            main.objects_created[instance_name] = obj
            return obj

    return inner


def __sub_map(pattern, f, s):
    def map_match(match):
        return f(match.group())

    return re.sub(pattern, map_match, s)


"""TODO:
WHAT IS THIS?
DESCRIBE WHAT IT IS AND WHAT IT DOES
NEED COMMENT SECTION
"""


def obtain(obj_repr):
    if not isinstance(obj_repr, str):
        obj_repr = str(obj_repr)
    obj_repr = (
        obj_repr.replace("__QUOTATION__", '"')
        .replace("__APOSTROPHE__", "'")
        .replace("__TILDE__", "~")
    )
    obj_repr = _Plugins__fake_file_config_source.numerize_string(obj_repr)
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

# TODO: PLEASE ALL IMPORT INTO THE HEADER
from .__hydra_main import main as hydramain


# TODO: DESCRIBE WHY THIS IS CALLED COMPLEMENTED. EXPLAIN THE IDEA BEHIND

class ComplementedConfig:
    """A config object, generated by ``rcognita``'s config pipeline."""

    def __init__(self, cfg, config_path=""):
        """Initialize a complemented config.

        :param cfg: hydra config object
        :param config_path: path to a config file
        :return: The self object
        :doc-author: Trelent
        """
        self._hydra_config = cfg
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
            str(self._hydra_config)
            .replace("DictConfig", "ComplementedConfigDict")
            .replace("ListConfig", "ComplementedConfigList")
            .replace("${get:", "={")
            .replace("${same:", "~{")
        )

    def __repr__(self):
        return self.__class__.__name__ + "( " + repr(self._hydra_config) + ")"

    def __hash__(self):
        if not self._saved_hash:
            real_config = omegaconf.OmegaConf.to_container(self._hydra_config)
            return hash_string(json.dumps(real_config, sort_keys=True))
        else:
            return self._saved_hash

    def copy(self):
        return self.__class__(
            self._hydra_config.copy(), config_path=self.config_path
        )

    def __invert__(self):
        """Instantiate the object described by the config (recursively).

        ``_target_`` has to be specified.
        """
        # if not self.config_path:
        return inst.instantiate(self._hydra_config, path=self.config_path)
        # else:
        #     instance_name = self.config_path.strip()
        #     if instance_name in self.objects_created:
        #         return self.objects_created[instance_name]
        #     else:
        #         res = inst.instantiate(self.__hydra_config, path=self.config_path)
        #         self.objects_created[instance_name] = res
        #         return res

class ComplementedConfigList(ComplementedConfig):
    """A config object, generated by ``rcognita``'s config pipeline that corresponds to a complemented version of `ListConfig` from omegaconf."""

    def __contains__(self, item):
        return item in self._hydra_config

    def __getitem__(self, key):
        cfg = object.__getattribute__(self, "_hydra_config")
        name = key
        item = cfg[name]
        if isinstance(item, DictConfig) or isinstance(item, ListConfig):
            if self.config_path:
                child_config_path = f"{self.config_path}[{name}]"
            else:
                raise ValueError("Top level list configs are not supported.")
            entity = ComplementedConfigList if isinstance(item, ListConfig) else ComplementedConfigDict
            item = entity(item, config_path=child_config_path)
        return item

    def __setitem__(self, key, value):
        self._hydra_config[key] = value

    def __delitem__(self, key):
        del self._hydra_config[key]

class ComplementedConfigDict(ComplementedConfig):
    """A config object, generated by ``rcognita``'s config pipeline that corresponds to a complemented version of `DictConfig` from omegaconf."""

    # TODO: NEED COMMENT SECTION
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
                #    parent_node_raw = omegaconf.OmegaConf.to_container(parent_node.__hydra_config)
                if not isinstance(name, int):
                    real_name = (name if name in parent_node_raw else name + "__IGNORE__").strip()
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
                            value._hydra_config
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

            # TODO: WHAT THIS COMMENT REFERS TO?
            # append attributes to root

        raw_self = omegaconf.OmegaConf.to_container(self._hydra_config)
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
        return item in self._hydra_config or item + "__IGNORE__" in self._hydra_config

    def __getattr__(self, item):
        cfg = object.__getattribute__(self, "_hydra_config")
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
            entity = ComplementedConfigList if isinstance(item, ListConfig) else ComplementedConfigDict
            attr = entity(attr, config_path=child_config_path)
        return attr

    def __getitem__(self, key):
        cfg = object.__getattribute__(self, "_hydra_config")
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
            entity = ComplementedConfigList if isinstance(item, ListConfig) else ComplementedConfigDict
            item = entity(item, config_path=child_config_path)
        return item

    def __setitem__(self, key, value):
        if key + "__IGNORE__" in self._hydra_config:
            self._hydra_config[key + "__IGNORE__"] = value
        else:
            self._hydra_config[key] = value

    def __setattr__(self, key, value):
        if key in ["config_path", "_saved_hash"]:
            object.__setattr__(self, key, value)
            return
        if hasattr(self, "_hydra_config"):
            if key + "__IGNORE__" in self._hydra_config:
                self._hydra_config.__setattr__(key + "__IGNORE__", value)
            else:
                self._hydra_config.__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    def has_key(self, key):
        return key in self._hydra_config or key + "__IGNORE__" in self._hydra_config

    def clear(self):
        self._hydra_config.clear()

    def update(self, *args, **kwargs):
        for subdict in args:
            for key, value in subdict.values():
                self[key] = value
        for key, value in kwargs.values():
            self[key] = value

    def keys(self):
        return [key.replace("__IGNORE__", "") for key in self._hydra_config.keys()]

    def items(self):
        return [
            (
                key.replace("__IGNORE__", ""),
                value
                if not isinstance(value, DictConfig)
                and not isinstance(value, ListConfig)
                else (ComplementedConfigDict if isinstance(value, DictConfig) else ComplementedConfigList)(
                    value,
                    config_path=key
                    if not self.config_path
                    else f"{self.config_path}.{key}",
                ),
            )
            for key, value in self._hydra_config.items()
        ]

    def values(self):
        return [
            (
                value
                if not isinstance(value, DictConfig)
                and not isinstance(value, ListConfig)
                else (ComplementedConfigDict if isinstance(value, DictConfig) else ComplementedConfigList)(
                    value,
                    config_path=key
                    if not self.config_path
                    else f"{self.config_path}.{key}",
                )
            )
            for key, value in self._hydra_config.items()
        ]

    def __delitem__(self, key):
        if key + "__IGNORE__" in self._hydra_config:
            del self._hydra_config[key + "__IGNORE__"]
        else:
            del self._hydra_config[key]



class RcognitaExitException(Exception):
    """Raised to forcefully shut down current simulation."""

    pass


# TODO: ADD 2-3 GENERAL SENTENCES OF WHAT THIS AND WHAT ITS FUNCTIONALITY ARE
class main:
    """The decorator used to invoke ``rcognita``'s config pipeline.

    Use it to decorate your ``main`` function like so:
    ::

         @rcognita.main(config_path=..., config_name=...)
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
        TimeCallback,
        ConfigDiagramCallback,
        TimeRemainingCallback,
        SaveProgressCallback,
    ]
    objects_created = {}

    class RcognitaArgumentParser(argparse.ArgumentParser):
        """A parser object designed for handling peculiar nuances of arg parsing while interfacing with hydra."""

        def __init__(self, *args, **kwargs):
            """Initialize and instance of RcognitaArgumentParser.

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
                if arg.startswith('--'):
                    key = arg[2:]
                elif arg.startswith('-'):
                    key = arg[1:]
                else:
                    raise ValueError("Argument name should start with either a '-' or a '--'.")
                key = key.replace('-', '_')
                if key == "help" or key == "h":
                    continue
                self._triggers[key] = trigger if trigger is not None else lambda v: None
            return res

        def parse_args(self):
            self.old_args = sys.argv
            self._stored_args = [arg for arg in sys.argv if arg.split("=")[0] in self._registered_args]
            res = super().parse_args(self._stored_args)
            sys.argv = [arg for arg in sys.argv if arg not in self._stored_args]
            for key in self._triggers:
                self._triggers[key](vars(res)[key])
            self._registered_args = []

            sys.argv.insert(1, "--multirun")
            sys.argv.insert(-1, "hydra.job.chdir=True")

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

    # TODO: DOCSTRING
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

    # TODO: DOCSTRING TO SHORT AND UNINFORMATIVE
    def __init__(
        self,
        *args,
        logger=None,
        callbacks=None,
        **kwargs,
    ):
        """Create an instance of ``rcognita.main``.

        :param config_path: path to the folder containing your config(s)
        :type config_path: str
        :param config_name: the name of your config (without extension)
        :type config_name: str
        :param callbacks: list of uninstantiated callbacks (classes) to be registered
        :type callbacks: list[class], optional
        :param logger: a logger instance to be supplied to callbacks
        :type logger: Logger, optional

        """
        if logger is None:
            logger = logging.getLogger("rcognita")
        if callbacks is None:
            callbacks = []

        self.callbacks_ = callbacks + self.builtin_callbacks

        self.parser = self.RcognitaArgumentParser(description="Run with config ...")

        self.parser.add_argument("--no-git", action='store_true')

        def disable_logging(flag):
            if flag:
                sys.argv.insert(-1, "hydra/job_logging=disabled")
        self.parser.add_argument("--disable-logging", action='store_true',
                                 trigger=disable_logging)

        def disable_callbacks(flag):
            if flag:
                self.__class__.builtin_callbacks = []
                self.callbacks_ = []
        self.parser.add_argument("--disable-callbacks", action='store_true', trigger=disable_callbacks)

        self.parser.add_argument("--enable-streamlit", action='store_true')

        self.parser.add_argument("--interactive", action='store_true')

        self.parser.add_argument("--cooldown-factor", default=1.0)

        def single_thread(flag):
            if not flag:
                sys.argv.insert(-1, "hydra/launcher=joblib")
        self.parser.add_argument("--single-thread", action='store_true', trigger=single_thread)

        def sweep(flag):
            if flag:
                sys.argv.insert(-1, "hydra/sweeper=ax")
                self.is_sweep = True
            else:
                self.is_sweep = False
        self.parser.add_argument("--sweep", action='store_true')


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

        self.mlflow_tracking_uri = f"file://{os.getcwd()}/mlruns"
        self.mlflow_artifacts_location = self.mlflow_tracking_uri + "/artifacts"

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
        self.__class__.logger = logger
        self.__class__.callbacks = [callback() for callback in self.callbacks_]

    # TODO: DESCRIBE WHAT THIS DOES
    def __call__(self, old_app):
        def rcognita_main(*args, **kwargs):
            argv = self.parser.parse_args()
            common_dir = tempfile.TemporaryDirectory()
            initial_working_directory = os.getcwd()
            initial_pythonpath = (
                os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""
            )
            script_path = inspect.getfile(old_app)
            no_git = argv.no_git
            path_main = os.path.abspath(script_path)
            path_parent = "/".join(path_main.split("/")[:-1])
            os.chdir(path_parent)
            path = os.path.abspath(self.kwargs["config_path"])
            self.kwargs["config_path"] = path

            def app(
                cfg,
                callbacks=self.__class__.callbacks,
                argv=argv,
                logger=self.__class__.logger,
                is_clear_matplotlib_cache_in_callbacks=self.__class__.is_clear_matplotlib_cache_in_callbacks,
            ):
                #args = self.parser.parse_args()
                self.__class__.is_clear_matplotlib_cache_in_callbacks = (
                    is_clear_matplotlib_cache_in_callbacks
                )
                self.__class__.logger = logger

                if "seed" in cfg:
                    seed = cfg["seed"]
                    delattr(cfg, "seed")
                else:
                    seed = 0

                if "mlflow_tracking_uri__IGNORE__" in cfg:
                    try:
                        mlflow_tracking_uri = cfg.mlflow_tracking_uri__IGNORE__
                    except:  # noqa: E722
                        mlflow_tracking_uri = self.mlflow_tracking_uri
                    delattr(cfg, "mlflow_tracking_uri__IGNORE__")
                    self.mlflow_tracking_uri = mlflow_tracking_uri

                mlflow.set_tracking_uri(self.mlflow_tracking_uri)

                if "mlflow_artifacts_location__IGNORE__" in cfg:
                    try:
                        artifacts_location = cfg.mlflow_artifacts_location__IGNORE__
                    except:  # noqa: E722
                        artifacts_location = self.mlflow_artifacts_location
                    delattr(cfg, "mlflow_artifacts_location__IGNORE__")
                    self.mlflow_artifacts_location = artifacts_location
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
                        "script_path": script_path,
                        "config_path": path + f"/{self.kwargs['config_name']}.yaml",
                        "initial_working_directory": initial_working_directory,
                        "initial_pythonpath": initial_pythonpath,
                        "common_dir": common_dir.name,
                        "id": int(os.getcwd().split("/")[-1]),
                        "report": lambda: shelve.open(
                            common_dir.name
                            + "/report_"
                            + os.getcwd().split("/")[-1].zfill(5)
                        ),
                        "pid": os.getpid(),
                        "argv": argv,
                        "main": main

                    }
                    callbacks[0]._metadata = self.__class__.metadata
                    ccfg = ComplementedConfigDict(cfg)
                    self.apply_assignments(ccfg)
                    if "callbacks" in cfg and not argv.disable_callbacks:
                        for callback in cfg.callbacks:
                            callback = (
                                obtain(callback)
                                if isinstance(callback, str)
                                else callback
                            )
                            callbacks.insert(-1, callback())
                        delattr(cfg, "callbacks")
                    elif "callbacks" in cfg:
                        delattr(cfg, "callbacks")
                    self.callbacks = callbacks
                    self.__class__.config = ccfg

                    try:
                        for callback in self.callbacks:
                            if callback.cooldown:
                                callback.cooldown *= argv.cooldown_factor
                            callback.on_launch()  # TODO: make sure this line is adequate to mlflow functionality
                        with self.__class__.metadata["report"]() as r:
                            r["path"] = os.getcwd()
                            r["pid"] = os.getpid()
                        self.tags.update({"run_path": os.getcwd()})

                        if mlflow.get_experiment_by_name(self.experiment_name) is None:
                            experiment_id = mlflow.create_experiment(
                                name=self.experiment_name,
                                artifact_location=self.mlflow_artifacts_location,
                            )
                        else:
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
                                for line in OmegaConf.load(".hydra/overrides.yaml")
                            }
                            mlflow.log_params(overrides)
                            mlflow.log_artifact("SUMMARY.html")
                            mlflow.log_artifact(".summary")
                            mlflow.log_artifact(".hydra")
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

                        self.__class__.callbacks[0].log(
                            "Script terminated successfully."
                        )
                    except RcognitaExitException as e:
                        res = e
                    except InterpolationResolutionError as e:
                        with self.__class__.metadata["report"]() as r:
                            r["traceback"] = traceback.format_exc()
                        res = e
                        self.__class__.callbacks[0].log("Script terminated with error. This error occurred when trying to instantiate something from the config.")
                        self.__class__.callbacks[0].exception(e)
                    except Exception as e:
                        with self.__class__.metadata["report"]() as r:
                            r["traceback"] = traceback.format_exc()
                        res = e
                        self.__class__.callbacks[0].log("Script terminated with error.")
                        self.__class__.callbacks[0].exception(e)
                    for callback in self.__class__.callbacks:
                        try:
                            callback.on_termination(res)
                        except Exception as e:
                            callback.log(
                                f"Termination procedure for {callback.__class__.__name__} failed."
                            )
                            callback.exception(e)
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
                        return {
                            "result": res,
                            "callbacks": self.__class__.callbacks,
                            "directory": os.getcwd(),
                        }

            # TODO: DOCSTRING
            def gui_server():
                import streamlit.web.bootstrap
                from streamlit import config as _config

                _config.set_option("server.headless", True)
                args = [common_dir.name]

                # streamlit.cli.main_run(filename, args)
                streamlit.web.bootstrap.run(gui_script_file, "", args, flag_options={})

            gui = Process(target=gui_server) if argv.enable_streamlit else Mock()
            gui.start()
            app.__module__ = old_app.__module__
            res = hydramain(*self.args, **self.kwargs)(app)(*args, **kwargs)
            common_dir.cleanup()
            time.sleep(2.0)
            gui.terminate()
            self.parser.restore_args()
            return res

        return rcognita_main


warnings.filterwarnings("ignore", category=UserWarning, module=hydra.__name__)
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

array = __utilities.rc.array
rc = __utilities.rc

class FancyModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return main(*args, **kwargs)

sys.modules[__name__].__class__ = FancyModule
