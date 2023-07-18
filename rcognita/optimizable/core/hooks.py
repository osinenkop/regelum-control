import rcognita
from typing import Callable, List
from types import GeneratorType


def get_data_hook(var):
    def var_hook(whatever):
        return var.data

    return var_hook


def requires_grad(variable_data):
    if isinstance(variable_data, (List, GeneratorType)):
        for datum in variable_data:
            if isinstance(datum, tuple):
                datum[1].requires_grad_(True)
            else:
                datum.requires_grad_(True)
    elif isinstance(variable_data, Callable):
        return requires_grad(variable_data())
    else:
        variable_data.requires_grad_(True)
    return variable_data


def detach(variable_data):
    if isinstance(variable_data, (List, GeneratorType)):
        for datum in variable_data:
            if isinstance(datum, tuple):
                datum[1].requires_grad_(False)
            else:
                datum.requires_grad_(False)
    elif isinstance(variable_data, Callable):
        return detach(variable_data())
    else:
        variable_data.requires_grad_(False)
    return variable_data


def mutate_metadata(self, new_metadata, tag="default"):
    def metadata_mutator(whatever):
        return new_metadata

    hook = rcognita.Hook(metadata_mutator, metadata=tag, act_on="metadata")
    return hook
