from abc import ABC, abstractmethod
import rcognita.systems as systems
import rcognita.actors as actors
import rcognita.controllers as controllers
import rcognita.critics as critics
import rcognita.models as models
import rcognita.objectives as objectives
import rcognita.observers as observers
import rcognita

def apply_callbacks(method):
    def new_method(self, *args, **kwargs):
        res = method(self, *args, **kwargs)
        if self.callbacks is None:
            self.callbacks = rcognita.main.callbacks
        for callback in self.callbacks:
            callback(obj=self, method=method.__name__, output=res)

    return new_method


class introduce_callbacks:
    def __init__(self, default_callbacks=None):
        self.default_callbacks = default_callbacks


    def __call__(self, cls):
        class whatever(cls):
            def __init__(self2, *args, callbacks=self.default_callbacks, **kwargs):
                super().__init__(*args, **kwargs)
                self2.callbacks = callbacks

        return whatever








class Callback(ABC):
    def __init__(self, logger, log_level="info"):
        self.log = logger.__getattribute__(log_level)

    @abstractmethod
    def perform(self, obj, method, output):
        pass


    def __call__(self, obj, method, output):
        self.performed_bases = []
        for base in self.__class__.__bases__:
            if base is not Callback and base not in self.peformed_bases:
                base(self, obj, method, output)
                self.performed_bases.append(base)
        self.perform(obj, method, output)


def method_callback(method_name, class_name=None, log_level="debug"):
    if class_name is not None:
        class_name = class_name.__name__ if not isinstance(class_name, str) else class_name
    class MethodCallback(Callback):
        def __init__(self, log, log_level=log_level):
            super().__init__(log, log_level=log_level)

        def perform(self, obj, method, output):
            if method==method_name and class_name in [None, obj.__class__.__name__]:
                self.log(f"Method '{method}' of class '{obj.__class__.__name__}' returned {output}")
    return MethodCallback


class StateCallback(Callback):
    def perform(self, obj, method, output):
        if isinstance(obj, systems.System) and method == systems.System.compute_closed_loop_rhs.__name__:
            self.log(f"System's state: {obj._state}")


class ObjectiveCallback(Callback):
    def perform(self, obj, method, output):
        if isinstance(obj, actors.Actor) and method == 'objective':
            self.log(f"Current objective: {output}")




