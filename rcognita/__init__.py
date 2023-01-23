__version__ = "0.2.1"

import sys, os, inspect
import logging
import warnings

import omegaconf

from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.resolvers import oc
import re

from typing import Any, Optional

from omegaconf import Container, Node
from omegaconf._utils import _DEFAULT_MARKER_, _get_value
from omegaconf.basecontainer import BaseContainer
from omegaconf.errors import ConfigKeyError
from omegaconf.grammar_parser import *

from recursive_monkey_patch import monkey_patch

import hydra.core.plugins
import hydra._internal.config_loader_impl
from . import _Plugins__fake_file_config_source
from . import __fake_plugins
from . import __fake_config_loader_impl

monkey_patch(__fake_plugins, hydra.core.plugins)
monkey_patch(__fake_config_loader_impl, hydra._internal.config_loader_impl)

from . import controllers
from . import systems
from . import simulator
from . import systems
from . import loggers
from .visualization import animator
from . import __utilities
from . import models
from . import predictors
from . import actors
from . import visualization

from unittest.mock import Mock, MagicMock
from hydra._internal.utils import _locate

from . import __instantiate as inst

mock = Mock()

objects_created = None


def __memorize_instance(resolver):
    global objects_created
    objects_created = {}

    def inner(
        key: str,
        default: Any = _DEFAULT_MARKER_,
        *,
        _parent_: Container,
    ):
        obj = inst.instantiate(resolver(key, default=default, _parent_=_parent_))
        if default == _DEFAULT_MARKER_:
            default = key.strip()
        instance_name = str(default)
        if instance_name in objects_created:
            return objects_created[instance_name]
        else:
            objects_created[instance_name] = obj
            return obj

    return inner


def __sub_map(pattern, f, s):
    def map_match(match):
        return f(match.group())

    return re.sub(pattern, map_match, s)


def __obtain(obj_repr):
    if not isinstance(obj_repr, str):
        obj_repr = str(obj_repr)
    obj_repr = (
        obj_repr.replace("__QUOTATION__", '"')
        .replace("__APOSTROPHE__", "'")
        .replace("__TILDE__", "~")
    )
    obj_repr = _Plugins__fake_file_config_source.numerize_string(obj_repr)
    pattern = re.compile(r"(\A|[^a-zA-Z\._])[a-zA-Z_][a-zA-Z0-9_]*")
    resolved = []

    def resolve(s):
        if s[0].isalnum() or s[0] == "_":
            prefix = ""
        else:
            prefix = s[0]
            s = s[1:]
        try:
            entity = _locate(s)
        except:
            entity = eval(s)
        resolved.append(entity)
        return f"{prefix}resolved[{len(resolved) - 1}]"

    obj_repr_resolved_modules = __sub_map(pattern, resolve, obj_repr)
    return eval(obj_repr_resolved_modules)


OmegaConf.register_new_resolver("same", __memorize_instance(oc.select))
OmegaConf.register_new_resolver(name="get", resolver=__obtain)
OmegaConf.register_new_resolver(name="mock", resolver=lambda: mock)


from .__hydra_main import main as hydramain


class ComplementedConfig:
    """
    A config object, generated by ``rcognita``'s config pipeline.
    """

    def __init__(self, cfg, config_path=""):
        self.__hydra_config = cfg
        self.config_path = config_path

    def refresh(self):
        """
        Reset all cached instances.
        Call this if you made changes to your config dynamically and would like those change to take
        effect in forthcoming instantiations.
        """
        objects_created.clear()

    def __getattr__(self, item):
        cfg = object.__getattribute__(self, "_ComplementedConfig__hydra_config")
        try:
            name = item
            attr = cfg.__getattr__(name)
        except (IndexError, KeyError, AttributeError) as e:
            name = item + "__IGNORE__"
            attr = cfg.__getattr__(name)
        if isinstance(attr, DictConfig) or isinstance(attr, ListConfig):
            if self.config_path:
                child_config_path = self.config_path + "." + name
            else:
                child_config_path = name
            attr = ComplementedConfig(attr, config_path=child_config_path)
        return attr

    def __str__(self):
        return (
            str(self.__hydra_config)
            .replace("DictConfig", "ComplementedConfig")
            .replace("ListConfig", "ComplementedConfig")
            .replace("${get:", "={")
            .replace("${same:", "~{")
        )

    def __getitem__(self, key):
        cfg = object.__getattribute__(self, "_ComplementedConfig__hydra_config")
        try:
            name = key
            item = cfg[name]
        except (IndexError, KeyError, AttributeError) as e:
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
            item = ComplementedConfig(item, config_path=child_config_path)
        return item

    def __setitem__(self, key, value):
        if key + "__IGNORE__" in self.__hydra_config:
            self.__hydra_config[key + "__IGNORE__"] = value
        else:
            self.__hydra_config[key] = value

    def __repr__(self):
        return self.__class__.__name__ + "( " + repr(self.__hydra_config) + ")"

    def __setattr__(self, key, value):
        if key == "config_path":
            object.__setattr__(self, key, value)
            return
        if hasattr(self, "_ComplementedConfig__hydra_config"):
            if key + "__IGNORE__" in self.__hydra_config:
                self.__hydra_config.__setattr__(key + "__IGNORE__", value)
            else:
                self.__hydra_config.__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    def copy(self):
        return ComplementedConfig(
            self.__hydra_config.copy(), config_path=self.config_path
        )

    def has_key(self, key):
        return key in self.__hydra_config or key + "__IGNORE__" in self.__hydra_config

    def clear(self):
        self.__hydra_config.clear()

    def update(self, *args, **kwargs):
        for subdict in args:
            for key, value in subdict.values():
                self[key] = value
        for key, value in kwargs.values():
            self[key] = value

    def keys(self):
        return [key.replace("__IGNORE__", "") for key in self.__hydra_config.keys()]

    def values(self):
        return [
            (
                key.replace("__IGNORE__", ""),
                value
                if not isinstance(value, DictConfig)
                and not isinstance(value, ListConfig)
                else ComplementedConfig(
                    value,
                    config_path=key
                    if not self.config_path
                    else f"{self.config_path}.{key}",
                ),
            )
            for key, value in self.__hydra_config.values()
        ]

    def __delitem__(self, key):
        if key + "__IGNORE__" in self.__hydra_config:
            del self.__hydra_config[key + "__IGNORE__"]
        else:
            del self.__hydra_config[key]

    def __invert__(self):
        """
        Instantiate the object described by the config (recursively).

        ``_target_`` has to be specified.
        """
        if not self.config_path:
            return inst.instantiate(self.__hydra_config, path=self.config_path)
        else:
            instance_name = self.config_path.strip()
            if instance_name in objects_created:
                return objects_created[instance_name]
            else:
                res = inst.instantiate(self.__hydra_config, path=self.config_path)
                objects_created[instance_name] = res
                return res


from .callbacks import *


class main:
    """
    The decorator used to invoke ``rcognita``'s config pipeline.

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
    assignments = []
    weak_assignments = []

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
        logger=logging.getLogger("rcognita"),
        callbacks=[StateCallback, ObjectiveCallback],
        **kwargs,
    ):
        """
        Create an instance of ``rcognita.main``.

        :param config_path: path to the folder containing your config(s)
        :type config_path: str
        :param config_name: the name of your config (without extension)
        :type config_name: str
        :param callbacks: list of uninstantiated callbacks (classes) to be registered
        :type callbacks: list[class], optional
        :param logger: a logger instance to be supplied to callbacks
        :type logger: Logger, optional

        """
        sys.argv.insert(1, "--multirun")
        sys.argv.insert(-1, "hydra.job.chdir=True")
        if "--disable-logging" in sys.argv:
            sys.argv.pop(sys.argv.index("--disable-logging"))
            sys.argv.insert(-1, "hydra/job_logging=disabled")
        self.cooldown_factor = 1.0
        for i, arg in enumerate(sys.argv):
            if "--cooldown-factor" in arg:
                self.cooldown_factor = float(arg.split("=")[-1])
                sys.argv.pop(i)
                break
        if not "--single-thread" in sys.argv:
            sys.argv.insert(-1, "hydra/launcher=joblib")
        else:
            sys.argv.pop(sys.argv.index("--single-thread"))
        if "--sweep" in sys.argv:
            sys.argv.insert(-1, "hydra/sweeper=ax")
            sys.argv.pop(sys.argv.index("--sweep"))
            self.is_sweep = True
        else:
            self.is_sweep = False
        self.args = args
        self.kwargs = kwargs
        self.kwargs["version_base"] = (
            None if "version_base" not in kwargs else kwargs["version_base"]
        )
        self.kwargs["config_path"] = (
            "." if "config_path" not in kwargs else kwargs["config_path"]
        )
        self.__class__.callbacks = [callback(logger) for callback in callbacks]
        self.__class__.logger = logger

    def __call__(self, old_app):
        def app(cfg, callbacks=self.__class__.callbacks):
            with omegaconf.flag_override(cfg, "allow_objects", True):
                ccfg = ComplementedConfig(cfg)
                self.apply_assignments(ccfg)
                if "callbacks" in cfg:
                    for callback in cfg.callbacks:
                        callback = (
                            __obtain(callback)
                            if isinstance(callback, str)
                            else callback
                        )
                        callbacks.append(callback(self.__class__.logger))
                    delattr(cfg, "callbacks")
                self.__class__.callbacks = callbacks
                for callback in self.__class__.callbacks:
                    if callback.cooldown:
                        callback.cooldown *= self.cooldown_factor
                res = old_app(ccfg)
                ccfg.refresh()
                if self.is_sweep:
                    return res
                else:
                    return {
                        "result": res,
                        "callbacks": self.__class__.callbacks,
                        "directory": os.getcwd(),
                    }

        app.__module__ = old_app.__module__
        path_main = os.path.abspath(inspect.getfile(old_app))
        path_parent = "/".join(path_main.split("/")[:-1])
        os.chdir(path_parent)
        path = os.path.abspath(self.kwargs["config_path"])
        self.kwargs["config_path"] = path
        return hydramain(*self.args, **self.kwargs)(app)


warnings.filterwarnings("ignore", category=UserWarning, module=hydra.__name__)


array = __utilities.rc.array
