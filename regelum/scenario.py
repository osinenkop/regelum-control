"""Contains various simulation scenarios.

For instance, an online scenario is when the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

"""


from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List
from unittest.mock import MagicMock

import regelum
from .simulator import Simulator
from .controller import Controller, RLController
from .objective import RunningObjective
from .constraint_parser import ConstraintParser, ConstraintParserTrivial
from .observer import Observer, ObserverTrivial
from . import ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK
from dataclasses import dataclass

try:
    import torch
except ImportError:
    torch = MagicMock()


def safe_round(precision=2, **kwargs):
    for k, v in kwargs.items():
        kwargs[k] = round(v, precision)
    return kwargs


class Scenario(regelum.RegelumBase, ABC):
    """A base scenario class."""

    def __init__(self):
        """Initialize an instance of Scenario."""
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def step(self):
        pass


# TODO: DOCSTRING
class OnlineScenario(Scenario):
    """Basic scenario for simulation and online learning."""

    @dataclass
    class AwaitedParameter:
        """A nested class to tag parameters that are expected to be computed at initial simulation step."""

        name: str

    @apply_callbacks()
    def __init__(
        self,
        simulator: Simulator,
        controller: Controller,
        running_objective: RunningObjective,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        howanim: Optional[str] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        speedup: int = 1,
        total_objective_threshold: float = np.inf,
        discount_factor: float = 1.0,
        observation_components_naming: Optional[List[str]] = None,
        action_components_naming: Optional[List[str]] = None,
    ):
        """Initialize an instance of OnlineScenario.

        This scenario is designed to run a simulation of a controlled system (a.k.a. an environment in RL).
        :param simulator: simulator in charge of said simulation
        :param controller: a controller that computes an action on each simulation step
        :param running_objective: an objective to be computed at each simulation step
        :param howanim: specifies animation mode
        :param state_init: initial state
        :param action_init: initial action
        :param time_start: time at which simulation starts
        :param observation_components_naming: names of each component of observation
        :param N_episodes: number of episodes in one iteration
        :param N_iterations: number of iterations in simulation
        :param speedup: number of frames to skip in order to speed up animation rendering
        """
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.simulator = simulator
        self.controller = controller
        self.running_objective = running_objective
        self.howanim = howanim
        self.is_playback = (
            self.howanim in ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK
        )
        self.total_objective = 0
        self.time = self.simulator.time_start
        self.time_old = 0
        self.delta_time = 0
        self.observation_components_naming = observation_components_naming
        self.action_components_naming = action_components_naming

        self.recent_total_objectives_of_episodes = []
        self.total_objectives_of_episodes = []
        self.total_objective_episodic_means = []

        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.current_scenario_status = "episode_continues"
        self.speedup = speedup
        self.total_objective_threshold = total_objective_threshold
        self.discount_factor = discount_factor
        self.is_episode_ended = False
        self.constraint_parser = (
            ConstraintParserTrivial()
            if constraint_parser is None
            else constraint_parser
        )
        self.observer = observer if observer is not None else ObserverTrivial()

        self.state_init, self.action_init = self.AwaitedParameter(
            "state_init"
        ), self.AwaitedParameter("action_init")
        self.state = self.state_init
        self.action = self.controller.action = self.action_init
        self.observation = self.AwaitedParameter("observation")

    def step(self):
        if isinstance(self.action_init, self.AwaitedParameter) and isinstance(
            self.state_init, self.AwaitedParameter
        ):
            (
                self.state_init,
                self.action_init,
            ) = self.simulator.get_init_state_and_action()
        is_total_objective_termination_criteria_satisfied = (
            self.total_objective > self.total_objective_threshold
        )
        if (
            not self.is_episode_ended
            and not is_total_objective_termination_criteria_satisfied
        ):
            (
                self.time,
                self.state,
                self.observation,
                self.simulation_metadata,
            ) = self.simulator.get_sim_step_data()

            self.delta_time = (
                self.time - self.time_old
                if self.time_old is not None and self.time is not None
                else 0
            )
            self.time_old = self.time
            self.constraint_parameters = self.constraint_parser.parse_constraints(
                simulation_metadata=self.simulation_metadata
            )
            self.controller.substitute_constraint_parameters(
                **self.constraint_parameters
            )
            state_estimated = self.observer.get_state_estimation(
                self.time, self.observation, self.action
            )

            self.action = self.controller.compute_action_sampled(
                self.time,
                state_estimated,
                self.observation,
            )
            self.simulator.receive_action(self.action)
            self.is_episode_ended = self.simulator.do_sim_step() == -1
            return "episode_continues"
        else:
            self.reset_episode()
            is_iteration_ended = self.episode_counter >= self.N_episodes

            if is_iteration_ended:
                self.reset_iteration()

                is_simulation_ended = self.iteration_counter >= self.N_iterations

                if is_simulation_ended:
                    self.reset_simulation()
                    return "simulation_ended"
                else:
                    return "iteration_ended"
            else:
                return "episode_ended"

    @apply_callbacks()
    def reload_pipeline(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.action = self.action_init
        self.simulator.reset()
        if isinstance(self.controller, RLController):
            self.recent_total_objective = self.controller.total_objective
            self.controller.policy.reset()
            self.controller.critic.reset()
            self.controller.reset()
        self.simulator.reset()
        self.observation = self.simulator.observation
        self.sim_status = 0
        return self.recent_total_objective

    def run(self):
        ### We use values `iteration_counter` and `episode_counter` only for debug purposes
        for _ in range(self.N_iterations):
            for _ in range(self.N_episodes):
                while self.sim_status not in [
                    "episode_ended",
                    "simulation_ended",
                    "iteration_ended",
                ]:
                    self.sim_status = self.step()

                self.reload_pipeline()

    @apply_callbacks()
    def reset_iteration(self):
        self.episode_counter = 0
        self.iteration_counter += 1
        self.recent_total_objectives_of_episodes = self.total_objectives_of_episodes
        self.total_objectives_of_episodes = []

        if self.sim_status != "simulation_ended":
            self.controller.optimize_on_event(event="reset_iteration")

    def reset_episode(self):
        self.episode_counter += 1
        self.is_episode_ended = False
        if self.sim_status != "simulation_ended":
            self.controller.optimize_on_event(event="reset_episode")

        return self.total_objective

    def reset_simulation(self):
        self.current_scenario_status = "episode_continues"
        self.iteration_counter = 0
        self.episode_counter = 0
