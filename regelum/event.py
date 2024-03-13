"""Contains Enum class representing different events that may occur inside the main loop."""

from enum import Enum, auto


class Event(Enum):
    """Enum representing different events that may occur inside the main loop.

    These events can be used to trigger specific actions or optimizations during the execution of the main loop.

    Attributes:
        reset_iteration: Indicates that an event occurs after resetting
            the iteration counter.
        reset_episode: Indicates that an event occurs after resetting
            the episode counter.
        compute_action: Indicates that an event occurs after computing
            an action.
        reset_simulation: Indicates that an event occurs after resetting
            the simulation.

    Usage:
        event = Event.reset_iteration
        if event == Event.reset_iteration:
            # Perform specific action or optimization after resetting the iteration counter
            pass
    """

    reset_iteration = auto()
    reset_episode = auto()
    compute_action = auto()
    reset_simulation = auto()
