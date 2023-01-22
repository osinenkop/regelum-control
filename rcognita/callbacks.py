"""
This module contains callbacks.
Callbacks are lightweight event handlers, used mainly for logging.

To make a method of a class trigger callbacks, one needs to
decorate the class (or its base class) with ``@introduce_callbacks()`` and then
also decorate the method with ``@apply_callbacks``.

Callbacks can be registered by simply supplying them in the respective keyword argument:
::

    @rcognita.main(callbacks=[...], ...)
    def my_app(config):
        ...

"""

from abc import ABC, abstractmethod

import rcognita
import pandas as pd


def apply_callbacks(method):
    """
    Decorator that applies a list of callbacks to a given method of an object.
    If the object has no list of callbacks specified, the default callbacks are used.

    :param method: The method to which the callbacks should be applied.
    """

    def new_method(self, *args, **kwargs):
        res = method(self, *args, **kwargs)
        if self.callbacks is None:
            self.callbacks = rcognita.main.callbacks
        for callback in self.callbacks:
            callback(obj=self, method=method.__name__, output=res)
        return res

    return new_method


class introduce_callbacks:
    """
    A class decorator that introduces a `callbacks` attribute to the decorated class.
    The `callbacks` attribute is a list of callbacks that can be applied to methods
    of instances of the decorated class (for instance via `@apply_callbacks`).
    """

    def __init__(self, default_callbacks=None):
        """
        Initializes the decorator.

        :param default_callbacks: A list of callbacks that will be used as the default
            value for the `callbacks` attribute of instances of the decorated class.
            If no value is specified, the `callbacks` attribute will be initialized to `None`, which will
            in turn make `@apply_callbacks` use default callbacks instead.
        """
        self.default_callbacks = default_callbacks

    def __call__(self, cls):
        class whatever(cls):
            def __init__(self2, *args, callbacks=self.default_callbacks, **kwargs):
                super().__init__(*args, **kwargs)
                self2.callbacks = callbacks

        return whatever


class Callback(ABC):
    """
    This is the base class for a callback object. Callback objects are used to perform some action or computation after a method has been called.
    """

    def __init__(self, logger, log_level="info"):
        """
        Initialize a callback object.

        :param logger: A logging object that will be used to log messages.
        :type logger: logging.Logger
        :param log_level: The level at which messages should be logged.
        :type log_level: str
        """
        self.log = logger.__getattribute__(log_level)

    @abstractmethod
    def perform(self, obj, method, output):
        pass

    def __call__(self, obj, method, output):
        self.performed_bases = []
        for base in self.__class__.__bases__:
            if ABC not in base.__bases__ and base not in self.peformed_bases:
                base(self, obj, method, output)
                self.performed_bases.append(base)
        self.perform(obj, method, output)


class HistoricalCallback(Callback, ABC):
    @property
    @abstractmethod
    def data(self):
        pass


def method_callback(method_name, class_name=None, log_level="debug"):
    """
    Creates a callback class that logs the output of a specific method of a class or any class.

    :param method_name: Name of the method to log output for.
    :type method_name: str
    :param class_name: (Optional) Name of the class the method belongs to. If not specified, the callback will log the output for the method of any class.
    :type class_name: str or class
    :param log_level: (Optional) The level of logging to use. Default is "debug".
    :type log_level: str

    """
    if class_name is not None:
        class_name = (
            class_name.__name__ if not isinstance(class_name, str) else class_name
        )

    class MethodCallback(Callback):
        def __init__(self, log, log_level=log_level):
            super().__init__(log, log_level=log_level)

        def perform(self, obj, method, output):
            if method == method_name and class_name in [None, obj.__class__.__name__]:
                self.log(
                    f"Method '{method}' of class '{obj.__class__.__name__}' returned {output}"
                )

    return MethodCallback


class StateCallback(Callback):
    """
    StateCallback is a subclass of the Callback class that logs the state of a system object when the compute_closed_loop_rhs method of the System class is called.

    Attributes:
    log (function): A function that logs a message at a specified log level.
    """

    def perform(self, obj, method, output):
        if (
            isinstance(obj, rcognita.systems.System)
            and method == rcognita.systems.System.compute_closed_loop_rhs.__name__
        ):
            self.log(f"System's state: {obj._state}")


class ObjectiveCallback(Callback):
    """
    A Callback class that logs the current objective value of an Actor instance.

    This callback is triggered whenever the Actor.objective method is called.

    Attributes:
    log (function): A logger function with the specified log level.
    """

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.actors.Actor) and method == "objective":
            self.log(f"Current objective: {output}")


class ObjectiveCallbackMultirun(HistoricalCallback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache = {}
        self.timeline = []
        self.num_launch = 1

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "post_step":
            self.log(
                f"Current objective: {output[0]}, observation: {output[1]}, action: {output[2]}, outcome: {output[3]}"
            )
            key = (self.num_launch, obj.time)
            if key in self.cache.keys():
                self.num_launch += 1
                key = (self.num_launch, obj.time)

            self.cache[key] = output[0]
            if self.timeline != []:
                if self.timeline[-1] < key[1]:
                    self.timeline.append(key[1])

            else:
                self.timeline.append(key[1])

    @property
    def data(self):
        keys = list(self.cache.keys())
        run_numbers = sorted(list(set([k[0] for k in keys])))
        cache_transformed = {key: list() for key in run_numbers}
        for k, v in self.cache.items():
            cache_transformed[k[0]].append(v)
        return cache_transformed


class TotalObjectiveCallbackMultirun(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = pd.DataFrame()

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "reload_pipeline":
            self.log(f"Current total objective: {output}")
            episode = (
                obj.episode_counter
                if obj.episode_counter != 0
                else self.cache.index.max() + 1
            )
            row = pd.DataFrame({"outcome": output}, index=[episode])
            self.cache = pd.concat([self.cache, row])

    @property
    def data(self):
        return self.cache.iloc[:-1]
