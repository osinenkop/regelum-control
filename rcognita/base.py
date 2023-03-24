import abc
import inspect
from .callbacks import Callback
import rcognita

class RcognitaBase(abc.ABC):
    def __init__(self):
        callbacks = [getattr(self.__class__, d) for d in dir(self.__class__)
                     if inspect.isclass(getattr(self.__class__, d)) and issubclass(getattr(self.__class__, d), Callback)]
        existing_callbacks = [type(callback) for callback in rcognita.main.callbacks]
        for callback in callbacks:
            if callback not in existing_callbacks:
                callback_instance = callback()
                callback_instance.on_launch()
                rcognita.main.callbacks = [callback_instance] + rcognita.main.callbacks
