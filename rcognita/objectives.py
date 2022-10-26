"""
Module that contains general objectives functions that can be used by various entities of the framework.
For instance, a running objective can be used commonly by a generic optimal controller, an actor, a critic, a logger, an animator, a pipeline etc.

"""

from .utilities import rc
from abc import ABC, abstractmethod
from torch.nn import Module
import actors


class Objective(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class RunningObjective(Objective):
    """
    This is what is usually treated as reward or unitlity in maximization problems.
    In minimzations problems, it is called cost or loss, say.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, observation, action):

        running_objective = self.model(observation, action)

        return running_objective
