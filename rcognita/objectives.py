"""
Module that contains general objectives functions that can be used by various entities of the framework.
For instance, a running objective can be used commonly by a generic optimal controller, an actor, a critic, a logger, an animator, a pipeline etc.

"""

from abc import ABC, abstractmethod

import rcognita.base
from .models import Model
from typing import Optional


class Objective(rcognita.base.RcognitaBase, ABC):
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

    # TODO: vypilit observation target
    def __init__(self, model: Optional[Model] = None, observation_target=None):
        """
        Initialize a RunningObjective instance.

        :param model: function that calculates the running objective for a given observation and action.
        :type model: function
        """
        self.observation_target = observation_target
        self.model = (lambda observation, action: 0) if model is None else model

    def __call__(self, observation, action):
        """
        Calculate the running objective for a given observation and action.

        :param observation: current observation.
        :type observation: numpy array
        :param action: current action.
        :type action: numpy array
        :return: running objective value.
        :rtype: float
        """

        if self.observation_target is not None:
            observation = observation - self.observation_target
        running_objective = self.model(observation, action)

        return running_objective
