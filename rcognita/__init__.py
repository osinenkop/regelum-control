__version__ = "v0.1.2"

import sys
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
import colored_traceback

from unittest.mock import Mock
from hydra._internal.utils import _locate

from . import __instantiate as inst

mock = Mock()

reset_instances = None
objects_created = None


def memorize_instance(resolver):
    global objects_created
    objects_created = {}
    global reset_instances

    def reset_instances():
        if objects_created:
            warnings.warn(
                "Object instantiations within your config have been reset. "
                "The objects that you instantiated from your config no "
                "longer refer to those that you are about to instantiate"
                "from respective config paths."
            )
        objects_created.clear()

    def inner(
        key: str, default: Any = _DEFAULT_MARKER_, *, _parent_: Container,
    ) -> Any:
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


def sub_map(pattern, f, s):
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

    obj_repr_resolved_modules = sub_map(pattern, resolve, obj_repr)
    return eval(obj_repr_resolved_modules)


OmegaConf.register_new_resolver("same", memorize_instance(oc.select))
OmegaConf.register_new_resolver(name="get", resolver=obtain)
OmegaConf.register_new_resolver(name="mock", resolver=lambda: mock)

colored_traceback.add_hook()

from hydra import main as hydramain


class ComplementedConfig:
    def __init__(self, cfg, config_path=""):
        self.__hydra_config = cfg
        self.config_path = config_path

    def refresh(self):
        reset_instances()

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
        self.args = args
        self.kwargs = kwargs
        self.kwargs["version_base"] = (
            None if "version_base" not in kwargs else kwargs["version_base"]
        )
        self.__class__.callbacks = [callback(logger) for callback in callbacks]
        self.__class__.logger = logger

    def __call__(self, old_app):
        def app(cfg):
            with omegaconf.flag_override(cfg, "allow_objects", True):
                ccfg = ComplementedConfig(cfg)
                self.apply_assignments(ccfg)
                if "callbacks" in cfg:
                    for callback in cfg.callbacks:
                        callback = (
                            obtain(callback) if isinstance(callback, str) else callback
                        )
                        self.__class__.callbacks.append(callback(self.__class__.logger))
                    delattr(cfg, "callbacks")
                return old_app(ccfg)

        app.__module__ = old_app.__module__
        return hydramain(*self.args, **self.kwargs)(app)


warnings.filterwarnings("ignore", category=UserWarning, module=hydra.__name__)


array = __utilities.rc.array
