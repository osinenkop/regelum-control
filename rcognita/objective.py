"""Module that contains general objectives functions that can be used by various entities of the framework.

For instance, a running objective can be used commonly by a generic optimal controller, an actor, a critic, a logger, an animator, a pipeline etc.

"""

from abc import ABC, abstractmethod

import rcognita.base
from .model import Model
from typing import Optional


class Objective(rcognita.base.RcognitaBase, ABC):
    """A base class for objective implementations."""

    def __init__(self):
        """Initialize an instance of Objective."""
        pass

    @abstractmethod
    def __call__(self):
        pass


class RunningObjective(Objective):
    """Running reward/cost.

    In minimzations problems, it is called cost or loss, say.
    """

    def __init__(self, model: Optional[Model] = None):
        """Initialize a RunningObjective instance.

        :param model: function that calculates the running objective for a given observation and action.
        :type model: function
        """
        self.model = (lambda observation, action: 0) if model is None else model

    def __call__(self, observation, action):
        """Calculate the running objective for a given observation and action.

        :param observation: current observation.
        :type observation: numpy array
        :param action: current action.
        :type action: numpy array
        :return: running objective value.
        :rtype: float
        """

        running_objective = self.model(observation, action)

        return running_objective
