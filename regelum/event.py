"""Contains Enum class representing different events that may occur inside the main loop."""

from enum import Enum, auto


class Event(Enum):
    """Enum representing different events that may occur inside the main loop.

    These events can be used to trigger specific actions or optimizations during the execution of the main loop.

    :ivar reset_iteration: Indicates that an event occurs after resetting the iteration counter.
    :ivar reset_episode: Indicates that an event occurs after resetting the episode counter.
    :ivar compute_action: Indicates that an event occurs after computing an action.
    :ivar reset_simulation: Indicates that an event occurs after resetting the simulation.

    Usage:
        event = Event.reset_iteration
        if event == Event.reset_iteration:
            # Perform specific action or optimization after resetting the iteration counter
            pass
    """

    reset_iteration = auto()
    reset_episode = auto
    compute_action = auto()
    reset_simulation = auto()
