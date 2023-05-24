import abc
import inspect
from .callbacks import Callback
import rcognita
import weakref


class RcognitaBase(abc.ABC):
    def __init__(self):
        callbacks = [
            getattr(self.__class__, d)
            for d in dir(self.__class__)
            if inspect.isclass(getattr(self.__class__, d))
            and issubclass(getattr(self.__class__, d), Callback)
        ]
        existing_callbacks = [type(callback) for callback in rcognita.main.callbacks]
        for callback in callbacks:
            if callback not in existing_callbacks:
                callback_instance = callback()
                callback_instance.on_launch()
                rcognita.main.callbacks = [callback_instance] + rcognita.main.callbacks


class Node(abc.ABC):
    def __init__(self, input_type):
        self.__subscribers = []
        self.__subscribees = []
        self.hooks = []
        self.type = input_type

    def hook(self, hook_function):
        self.hooks.append(hook_function)

    def __forget(self, other):
        assert self.connected(
            other
        ), "Attempt to disconnect a node that was not connected."
        for i, subscriber_ref in enumerate(self.__subscribers):
            subscriber = subscriber_ref()
            if subscriber is other:
                self.__subscribers.pop(i)
                break
        for i, subscribee_ref in enumerate(self.__subscribees):
            subscribee = subscribee_ref()
            if subscribee is other:
                self.__subscribees.pop(i)
                break

    def connected(self, other):
        return other in self.subscribees or other in self.subscribers

    def disconnect(self, other):
        self.__forget(other)
        other.__forget(self)

    def __del__(self):
        for subscriber in self.subscribers:
            self.disconnect(subscriber)
        for subscribee in self.subscribees:
            self.disconnect(subscribee)

    def __subscribe(self, other):
        assert isinstance(
            other, Node
        ), "Attempt to subscribe to something that is neither a Port nor a Publisher."
        assert issubclass(
            other.type, self.type
        ), f"Type mismatch. Attempt to subscribe a node of type {self.type} to node of type {other.type}."
        self.__subscribees.append(weakref.ref(other))
        other.__subscribers.append(weakref.ref(self))

    def __issue_subscription(self, other):
        assert isinstance(
            other, Node
        ), "Attempt to issue subscription to something that is neither a Port nor a Publisher."
        assert issubclass(
            self.type, other.type
        ), f"Type mismatch. Attempt to subscribe a node of type {other.type} to node of type {self.type}."
        self.__subscribers.append(weakref.ref(other))
        other.__subscribees.append(weakref.ref(self))

    @property
    def subscribers(self):
        return [
            subscriber()
            for subscriber in self.__subscribers
            if subscriber() is not None
        ]

    @property
    def subscribees(self):
        return [
            subscribee()
            for subscribee in self.__subscribees
            if subscribee() is not None
        ]

    @abc.abstractmethod
    def connect(self, other):
        pass

    @abc.abstractmethod
    def __on_input(self, message):
        pass

    def __call__(self, message):
        assert isinstance(
            message, self.type
        ), f"Type mismatch. Attempt to pass a message of type {type(message)} to node of type {self.type}."
        for hook in self.hooks:
            message = hook(message)
        return self.__on_input(message)


class EmptyInboxException(Exception):
    pass


class port:
    def __init__(self, input_type=object, hooks=None):
        if hooks is None:
            hooks = []
        self.input_type = input_type
        self.hooks = hooks

    def __call__(self, method):
        new_port = Port(self.input_type)
        new_port.handle(method)
        for hook in self.hooks:
            new_port.hook(hook)
        return new_port


class publisher:
    def __init__(self, input_type=object, hooks=None):
        if hooks is None:
            hooks = []
        self.input_type = input_type
        self.hooks = hooks

    def __call__(self, method):
        new_port = Publisher(self.input_type)
        new_port.hook(method)
        for hook in self.hooks:
            new_port.hook(hook)
        return new_port


class Port(Node):
    def __init__(self, input_type):
        super().__init__(input_type)
        self.__inbox = []
        self.__handler = None

    def connect(self, other):
        self.__subscribe(other)

    def __on_input(self, message):
        for subscriber in self.subscribers:
            subscriber(message)
        if self.__handler is None:
            self.__inbox.append(message)
        else:
            self.__handler(message)

    def handle(self, handler):
        self.__handler = handler

    def receive(self):
        assert self.handler is None, "``receive`` called on a handled port."
        if self.__inbox:
            return self.__inbox.pop(-1)
        else:
            raise EmptyInboxException(
                "The port's inbox was empty at the time of calling ``receive``."
            )


class Publisher(Node):
    def connect(self, other):
        self.__issue_subscription(other)

    def __on_input(self, message):
        for subscriber in self.subscribers:
            subscriber(message)


class FreePort(Port):
    def __init__(self):
        super().__init__(object)


class FreePublisher(Publisher):
    def __init__(self):
        super().__init__(object)


"""
class LazyMessageNotYetReceivedException(Exception):
    pass
class LazyMessage:
    def __init__(self, container, index, message_type):
        self.type = message_type
        self.container = container
        self.index = index

    def __str__(self):
        try:
            content = str(self())
        except LazyMessageNotYetReceivedException:
            content = "N/A"
        return f"{self.__class__.__name__}({content})"

    def __repr__(self):
        try:
            content = repr(self())
        except LazyMessageNotYetReceivedException:
            content = "N/A"
        return f"{self.__class__.__name__}({content})"

    def __call__(self):
        if len(self.container) < self.index:
            return self.container[self.index]
        else:
            raise LazyMessageNotYetReceivedException("Attempted to unbox a lazy message, but the message was not yet received.")

    def __add__(self, other):
        if isinstance(other, LazyMessage):


class LazyPort(Port):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = 0

    def receive(self):
        assert self.handler is None, "``receive`` called on a handled port."
        message = LazyMessage(container=self.__inbox, index=self.index, message_type=self.type)
        self.index += 1
        return message
"""
