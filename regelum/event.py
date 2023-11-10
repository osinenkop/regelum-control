from enum import Enum, auto


class BufferNullifyEvent(Enum):
    """Event indicating that the buffer should be nullified."""

    reset_iteration = auto()
    reset_episode = auto
    compute_action = auto()


class OptimizationEvent(Enum):
    """Event indicating that some `Optimizable` object should be optimized."""

    reset_iteration = auto()
    reset_episode = auto
    compute_action = auto()
