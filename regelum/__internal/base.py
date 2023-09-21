"""Base infrastructure of regelum."""

import abc

from regelum import callback as cb

import weakref



class apply_callbacks:
    """Decorator that applies a list of callbacks to a given method of an object.

    If the object has no list of callbacks specified, the default callbacks are used.

    :param method: The method to which the callbacks should be applied.
    """

    def __init__(self, callbacks=None):
        """Initialize a decorator that applies callbacks.

        :param callbacks: list of callbacks to apply (applies all default callbacks if omitted)
        """
        self.callbacks = callbacks

    def __call__(self, method):
        def new_method(self2, *args, **kwargs):
            res = method(self2, *args, **kwargs)
            if self.callbacks is None:
                callbacks = RegelumBase._metadata["main"].callbacks
            for callback in callbacks:
                callback(obj=self2, method=method.__name__, output=res)
            return res

        return new_method


class RegelumType(abc.ABCMeta):
    """Regelum type that all classes in regelum share.

    Used for certain infrastructural and syntactic sugar features.
    """

    @classmethod
    def __register_callback(cls, callback):
        pass

    @classmethod
    def __prepare__(metacls, *args):
        return {"apply_callbacks": apply_callbacks} | super().__prepare__(*args)

    def __add__(self, other):
        assert hasattr(self, "_compose")
        return self._compose(other)

    def __new__(cls, *args, **kwargs):
        x = super().__new__(cls, *args, **kwargs)
        if (
            x.__name__ == "RegelumBase"
            or x.__name__ == "Callback"
            or issubclass(x, cb.Callback)
        ):
            return x
        if hasattr(x, "_real_name"):
            x.__name__ = x._real_name
            del x._real_name
        callbacks = x._attached if hasattr(x, "_attached") else []
        if callbacks:
            del x._attached
        old_init = x.__init__

        def new_init(self, *args, **kwargs):
            for callback in callbacks:
                callback.register(attachee=x, launch=True)
            return old_init(self, *args, **kwargs)

        x.__init__ = new_init
        return x


class ClassPropertyDescriptor(object):
    """Enables to declare class properties."""

    def __init__(self, fget, fset=None):
        """Initialize an instance of ClassPropertyDescriptor.

        :param fget: class getter
        :param fset: class setter
        """
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.fget.__get__(obj, cls)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    """Decorate a method in such a way that it becomes a class property.

    :param func: method to decorate
    :return:
    """
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)





class RegelumBase(metaclass=RegelumType):
    """Base class designed to act as an abstraction over all regelum objects."""

    @classproperty
    def _metadata(self):
        return RegelumBase.__metadata

    @_metadata.setter
    def _metadata(self, metadata):
        if not hasattr(RegelumBase, f"_{RegelumBase.__name__}__metadata"):
            RegelumBase.__metadata = metadata
        else:
            raise ValueError(
                "Metadata has already been set, yet an attempt to set it again was made."
            )

    def __init__(self):
        """Initialize an object from regelum."""
        # if hasattr(self.__class__, "_attached"):
        #    existing_callbacks = [type(callback) for callback in regelum.main.callbacks]
        #    for callback in self._attached:
        #        if callback not in existing_callbacks:
        #            callback_instance = callback(attachee=self.__class__)  ## Might want to move it to the metaclass
        #            callback_instance.on_launch()  # I must change this
        #            regelum.main.callbacks = [callback_instance] + regelum.main.callbacks

        # callbacks = self.__class__._attached if hasattr(self.__class__, "_attached") else []
        # existing_callbacks = [type(callback) for callback in regelum.main.callbacks]
        # for callback in callbacks:
        #    if callback not in existing_callbacks:
        #        callback_instance = callback(attachee=self.__class__) ## Might want to move it to the metaclass
        #        callback_instance.on_launch()
        #        regelum.main.callbacks = [callback_instance] + regelum.main.callbacks







# TODO: DOCSTRING
class Node(abc.ABC):
    """A node is an object that is responsible for sending/receiving/rerouting/processing messages."""

    def __init__(self, input_type):
        """Initialize a node.

        :param input_type: type of messages sent/received by the node
        :type input_type: type
        """
        self.__subscribers = []
        self.__subscribees = []
        self.hooks = []
        self.type = input_type

    # TODO: DOCSTRING
    def hook(self, hook_function):
        def new_hook_function(input_):
            res = hook_function(input_)
            assert isinstance(
                res, self.type
            ), f"Values returned by hooks should match the type of their node (port/publisher). An object of type {type(res)} was returned, which does not match the node's type {self.type}."
            return res

        self.hooks.append(new_hook_function)

    # TODO: DOCSTRING
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

    # TODO: DOCSTRING
    def connected(self, other):
        return other in self.subscribees or other in self.subscribers

    # TODO: DOCSTRING
    def disconnect(self, other):
        self.__forget(other)
        other.__forget(self)

    # TODO: DOCSTRING
    def __del__(self):
        for subscriber in self.subscribers:
            self.disconnect(subscriber)
        for subscribee in self.subscribees:
            self.disconnect(subscribee)

    # TODO: DOCSTRING
    def __subscribe(self, other):
        assert isinstance(
            other, Node
        ), "Attempt to subscribe to something that is neither a Port nor a Publisher."
        assert issubclass(
            other.type, self.type
        ), f"Type mismatch. Attempt to subscribe a node of type {self.type} to node of type {other.type}."
        self.__subscribees.append(weakref.ref(other))
        other.__subscribers.append(weakref.ref(self))

    # TODO: DOCSTRING
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


# TODO: DOCSTRING
class EmptyInboxException(Exception):
    """Raised when trying to receive a message from a Port, to which no messages were yet sent."""

    pass


# TODO: DOCSTRING
class port:
    """Decorator factory that replaces a method with a Port in such a way that the initial definition of the method is interpreted as said Port's handler."""

    def __init__(self, input_type=object, hooks=None):
        """Initialize a decorator that transforms methods into handled ports.

        :param input_type: type of messages accepted by the resulting port
        :type input_type: type
        :param hooks: hooks to add in addition to the handler obtained from the decorated method
        """
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
    """Decorator factory that replaces a method with a Publisher in such a way that the initial definition of the method is interpreted as said Publisher's hook."""

    def __init__(self, input_type=object, hooks=None):
        """Initialize a decorator that transforms methods into hooked publishers.

        :param input_type: type of messages sent by the resulting publisher
        :type input_type: type
        :param hooks: hooks to add in addition to the hook obtained from the decorated method
        """
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
    """A Port is a Node that accepts messages."""

    def __init__(self, input_type):
        """Initialize a Port.

        :param input_type: type of messages received by the port.
        :type input_type: type
        """
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


# TODO: DOCSTRING
class Publisher(Node):
    """A Publisher is a Node that sends messages."""

    def connect(self, other):
        self.__issue_subscription(other)

    def __on_input(self, message):
        for subscriber in self.subscribers:
            subscriber(message)


class FreePort(Port):
    """A Port that can accept messages of arbitrary type."""

    def __init__(self):
        """Initialize a free port."""
        super().__init__(object)


class FreePublisher(Publisher):
    """A Publisher that can send messages of arbitrary type."""

    def __init__(self):
        """Initialize a free publisher."""
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
