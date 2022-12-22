from abc import ABC, abstractmethod
import rcognita.systems as systems
import rcognita.actors as actors
import rcognita.controllers as controllers
import rcognita.critics as critics
import rcognita.models as models
import rcognita.objectives as objectives
import rcognita.observers as observers
import rcognita

class Callback(ABC):
    def __init__(self, logger):
        self.log = logger

    @abstractmethod
    def perform(self, obj, method, output):
        pass


    def __call__(self, obj, method, output):
        for base in self.__class__.__bases__:
            if base is not Callback:
                base.perform(self, obj, method, output)
        self.perform(obj, method, output)



class StateCallback(Callback):
    def perform(self, obj, method, output):
        if isinstance(obj, systems.System) and method == systems.System.compute_closed_loop_rhs.__name__:
            self.log.info(f"System's state: {obj._state}")


class ObjectiveCallback(Callback):
    def perform(self, obj, method, output):
        if isinstance(obj, actors.Actor) and method == 'objective':
            self.log.info(f"Current objective: {output}")




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
