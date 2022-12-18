__version__ = "v0.1.2"

import omegaconf

from omegaconf import DictConfig, OmegaConf
from omegaconf.resolvers import oc

from typing import Any, Optional

from omegaconf import Container, Node
from omegaconf._utils import _DEFAULT_MARKER_, _get_value
from omegaconf.basecontainer import BaseContainer
from omegaconf.errors import ConfigKeyError
from omegaconf.grammar_parser import parse
from omegaconf.resolvers.oc import dict

import hydra
from hydra.utils import get_class, instantiate


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
import numpy
from unittest.mock import Mock
from hydra._internal.utils import _locate


mock = Mock()


def memorize_instance(resolver):
    objects_created = {}

    def inner(
        key: str, default: Any = _DEFAULT_MARKER_, *, _parent_: Container,
    ) -> Any:
        obj = instantiate(resolver(key, default=default, _parent_=_parent_))
        if obj.__class__.__name__ in objects_created:
            return objects_created[obj.__class__.__name__]
        else:
            objects_created[obj.__class__.__name__] = obj
            return obj

    return inner


def obtain(obj_repr):
    obj_repr = obj_repr.replace(")", ",)")
    if "(" in obj_repr:
        i = obj_repr.find("(")
        func = _locate(obj_repr[:i])
        return func(*eval(obj_repr[i:]))
    else:
        return _locate(obj_repr)


# OmegaConf.register_new_resolver(name="get_cls", resolver=lambda cls: get_class(cls))
OmegaConf.register_new_resolver("same", memorize_instance(oc.select))
OmegaConf.register_new_resolver(name="get", resolver=obtain)
OmegaConf.register_new_resolver(name="mock", resolver=lambda: mock)
# OmegaConf.register_new_resolver(name="rcognita", resolver=lambda var: get_class(f"rcognita.{var}"))

colored_traceback.add_hook()


class main:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, old_app):
        def app(cfg):
            with omegaconf.flag_override(cfg, "allow_objects", True):
                return old_app(cfg)

        app.__module__ = old_app.__module__

        return hydra.main(*self.args, **self.kwargs)(app)

