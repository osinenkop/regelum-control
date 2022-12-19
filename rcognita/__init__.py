__version__ = "v0.1.2"

import sys

import omegaconf

from omegaconf import DictConfig, OmegaConf
from omegaconf.resolvers import oc
import re

from typing import Any, Optional

from omegaconf import Container, Node
from omegaconf._utils import _DEFAULT_MARKER_, _get_value
from omegaconf.basecontainer import BaseContainer
from omegaconf.errors import ConfigKeyError
from omegaconf.grammar_parser import *
from omegaconf.grammar_parser import _grammar_cache
from omegaconf.resolvers.oc import dict

import antlr4
from . import __fakeantlr4
from recursive_monkey_patch import monkey_patch

monkey_patch(__fakeantlr4, antlr4)


from hydra.utils import instantiate

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







mock = Mock()


def memorize_instance(resolver):
    objects_created = {}

    def inner(
        key: str, default: Any = _DEFAULT_MARKER_, *, _parent_: Container,
    ) -> Any:
        obj = instantiate(resolver(key, default=default, _parent_=_parent_))
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
    pattern = re.compile(r'(\A|\s|\(|\[)[a-zA-Z]+')
    resolved = []
    def resolve(s):
        if s[0].isalnum():
            prefix = ""
        else:
            prefix = s[0]
            s = s[1:]
        try:
            entity = _locate(s)
        except:
            entity = eval(s)
        resolved.append(entity)
        return f"{prefix}resolved[{len(resolved) -  1}]"
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
        return ComplementedConfigWrapper(cfg.__getattr__(item))

    def __str__(self):
        return str(self.cfg)

    def __repr__(self):
        return self.__class__.__name__ + ": " + repr(self.cfg)

    def __setattr__(self, key, value):
        if hasattr(self, "cfg"):
            self.cfg.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def __invert__(self):
        return instantiate(self.cfg)



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

