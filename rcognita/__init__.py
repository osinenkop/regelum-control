__version__ = "v0.1.2"

import sys

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


def memorize_instance(resolver):
    objects_created = {}

    def inner(
            key: str, default: Any = _DEFAULT_MARKER_, *, _parent_: Container,
    ) -> Any:
        obj = inst.instantiate(resolver(key, default=default, _parent_=_parent_))
        key = obj.__class__.__name__ + str(default)
        if key in objects_created:
            return objects_created[key]
        else:
            objects_created[key] = obj
            return obj

    return inner


def sub_map(pattern, f, s):
    def map_match(match):
        return f(match.group())

    return re.sub(pattern, map_match, s)


def obtain(obj_repr):
    obj_repr = obj_repr.replace('__QUOTATION__', '"').replace("__APOSTROPHE__", "'")
    obj_repr = _Plugins__fake_file_config_source.numerize_string(obj_repr)
    pattern = re.compile(r"(\A|[^a-zA-Z\._])[a-zA-Z_]+")
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


class ComplementedConfigWrapper:
    def __init__(self, cfg):
        self.cfg = cfg

    def __getattr__(self, item):
        cfg = object.__getattribute__(self, "cfg")
        try:
            attr = cfg.__getattr__(item)
        except (IndexError, KeyError, AttributeError) as e:
            attr = cfg.__getattr__(item + "__IGNORE__")
        if isinstance(attr, DictConfig) or isinstance(attr, ListConfig):
            attr = ComplementedConfigWrapper(attr)
        return attr

    def __str__(self):
        return str(self.cfg)

    def __getitem__(self, key):
        cfg = object.__getattribute__(self, "cfg")
        try:
            item = cfg[key]
        except (IndexError, KeyError, AttributeError) as e:
            if isinstance(key, str):
                item = cfg[key + "__IGNORE__"]
            else:
                raise e
        if isinstance(item, DictConfig) or isinstance(item, ListConfig):
            attr = ComplementedConfigWrapper(item)
        return item

    def __setitem__(self, key, value):
        if key + "__IGNORE__" in self.cfg:
            self.cfg[key + "__IGNORE__"] = value
        else:
            self.cfg[key] = value

    def __repr__(self):
        return self.__class__.__name__ + ": " + repr(self.cfg)

    def __setattr__(self, key, value):
        if hasattr(self, "cfg"):
            if key + "__IGNORE__" in self.cfg:
                self.cfg.__setattr__(key + "__IGNORE__", value)
            else:
                self.cfg.__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    def copy(self):
        return ComplementedConfigWrapper(self.cfg.copy())

    def has_key(self, key):
        return key in self.cfg or key + "__IGNORE__" in self.cfg

    def clear(self):
        self.cfg.clear()

    def update(self, *args, **kwargs):
        for subdict in args:
            for key, value in subdict.values():
                self[key] = value
        for key, value in kwargs.values():
            self[key] = value

    def keys(self):
        return [key.replace("__IGNORE__", "") for key in self.cfg.keys()]

    def values(self):
        return [(key.replace("__IGNORE__", ""),
                 value if not isinstance(value, DictConfig) and not isinstance(value, ListConfig)
                 else ComplementedConfigWrapper(value))
                for key, value in self.cfg.values()]

    def __delitem__(self, key):
        if key + '__IGNORE__' in self.cfg:
            del self.cfg[key + '__IGNORE__']
        else:
            del self.cfg[key]

    def __invert__(self):
        return inst.instantiate(self.cfg)


class main:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, old_app):
        def app(cfg):
            with omegaconf.flag_override(cfg, "allow_objects", True):
                return old_app(ComplementedConfigWrapper(cfg))

        app.__module__ = old_app.__module__
        return hydramain(*self.args, **self.kwargs)(app)
