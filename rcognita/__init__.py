__version__ = "v0.1.2"

import sys
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
from . import _Plugins__fake_file_config_source
from . import __fake_plugins

monkey_patch(__fake_plugins, hydra.core.plugins)

from . import controllers
from . import systems
from . import simulator
from . import systems
from . import loggers
from .visualization import animator
from . import utilities
from . import models
from . import predictors
from . import actors
import colored_traceback

from unittest.mock import Mock
from hydra._internal.utils import _locate

from . import __instantiate as inst

mock = Mock()

reset_instances = None

def memorize_instance(resolver):
    objects_created = {}
    global reset_instances
    reset_instances = lambda: objects_created.clear()

    def inner(
            key: str, default: Any = _DEFAULT_MARKER_, *, _parent_: Container,
    ) -> Any:
        obj = inst.instantiate(resolver(key, default=default, _parent_=_parent_))
        if default == _DEFAULT_MARKER_:
            default = key.strip()
        instance_name = obj.__class__.__name__ + str(default)
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
    obj_repr = obj_repr.replace('__QUOTATION__', '"').replace("__APOSTROPHE__", "'").replace("__TILDE__", '~')
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
    def __init__(self, cfg):
        self.__hydra_config = cfg

    def refresh(self):
        reset_instances()

    def __getattr__(self, item):
        cfg = object.__getattribute__(self, "_ComplementedConfig__hydra_config")
        try:
            attr = cfg.__getattr__(item)
        except (IndexError, KeyError, AttributeError) as e:
            attr = cfg.__getattr__(item + "__IGNORE__")
        if isinstance(attr, DictConfig) or isinstance(attr, ListConfig):
            attr = ComplementedConfig(attr)
        return attr

    def __str__(self):
        return str(self.__hydra_config)\
            .replace("DictConfig", "ComplementedConfig")\
            .replace("ListConfig", "ComplementedConfig")\
            .replace("${get:", "={")\
            .replace("${same:", "~{}")

    def __getitem__(self, key):
        cfg = object.__getattribute__(self, "_ComplementedConfig__hydra_config")
        try:
            item = cfg[key]
        except (IndexError, KeyError, AttributeError) as e:
            if isinstance(key, str):
                item = cfg[key + "__IGNORE__"]
            else:
                raise e
        if isinstance(item, DictConfig) or isinstance(item, ListConfig):
            attr = ComplementedConfig(item)
        return item

    def __setitem__(self, key, value):
        if key + "__IGNORE__" in self.__hydra_config:
            self.__hydra_config[key + "__IGNORE__"] = value
        else:
            self.__hydra_config[key] = value

    def __repr__(self):
        return self.__class__.__name__ + "( " + repr(self.__hydra_config) + ")"

    def __setattr__(self, key, value):
        if hasattr(self, "_ComplementedConfig__hydra_config"):
            if key + "__IGNORE__" in self.__hydra_config:
                self.__hydra_config.__setattr__(key + "__IGNORE__", value)
            else:
                self.__hydra_config.__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    def copy(self):
        return ComplementedConfig(self.__hydra_config.copy())

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
        return [(key.replace("__IGNORE__", ""),
                 value if not isinstance(value, DictConfig) and not isinstance(value, ListConfig)
                 else ComplementedConfig(value))
                for key, value in self.__hydra_config.values()]

    def __delitem__(self, key):
        if key + '__IGNORE__' in self.__hydra_config:
            del self.__hydra_config[key + '__IGNORE__']
        else:
            del self.__hydra_config[key]

    def __invert__(self):
        return inst.instantiate(self.__hydra_config)


class main:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, old_app):
        def app(cfg):
            with omegaconf.flag_override(cfg, "allow_objects", True):
                return old_app(ComplementedConfig(cfg))

        app.__module__ = old_app.__module__
        return hydramain(*self.args, **self.kwargs)(app)

warnings.filterwarnings("ignore", category=UserWarning, module=hydra.__name__)
