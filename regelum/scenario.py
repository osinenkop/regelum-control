"""Contains various simulation scenarios.

For instance, an online scenario is when the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

"""


from abc import ABC, abstractmethod
from itertools import islice, cycle
import numpy as np
from typing import Optional, List
from unittest.mock import MagicMock

import regelum
from .__utilities import rc
from .simulator import Simulator
from .controller import Controller, RLController
from .objective import RunningObjective
from . import ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK

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

    cache = dict()

    @apply_callbacks()
    def __init__(
        self,
        simulator: Simulator,
        controller: Controller,
        running_objective: RunningObjective,
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

        This scenario is designed to run a simulation that involves an online learner.
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
        self.cache.clear()
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

        self.state_init, self.action_init = self.simulator.get_init_state_and_action()
        self.state = self.state_init
        self.action = self.controller.action = self.action_init
        self.observation = self.simulator.system.get_observation(
            self.time, self.state, self.action
        )

    def set_speedup(self, speedup):
        self.speedup = speedup
        self.cached_timeline = islice(cycle(iter(self.cache)), 0, None, self.speedup)

    def get_speeduped_cache_len(self):
        return len(range(0, len(self.cache), self.speedup))

    def get_cache_len(self):
        return len(self.cache)

    def update_total_objective(self, observation, action, delta):
        """Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.

        If the agent succeeded to stabilize the system, ``outcome`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize objective instead).
        """
        self.total_objective += (
            np.exp(self.time * np.log(self.discount_factor))
            * self.running_objective(observation, action)
            * delta
        )

    @apply_callbacks()
    def reload_pipeline(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.recent_total_objective = self.total_objective
        self.total_objective = 0
        self.action = self.action_init
        self.simulator.reset()
        if isinstance(self.controller, RLController):
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
        self.recent_total_objective = self.total_objective
        self.reset_episode()
        self.reset_iteration()
        self.reset_simulation()
        return self.recent_total_objective

    @apply_callbacks()
    def reset_iteration(self):
        self.episode_counter = 0
        self.iteration_counter += 1
        self.recent_total_objectives_of_episodes = self.total_objectives_of_episodes
        self.total_objectives_of_episodes = []

        if self.current_scenario_status != "simulation_ended":
            self.controller.optimize_on_event(event="reset_iteration")

    def reset_episode(self):
        self.total_objectives_of_episodes.append(self.total_objective)
        self.episode_counter += 1
        self.is_episode_ended = False
        if self.current_scenario_status != "simulation_ended":
            self.controller.optimize_on_event(event="reset_episode")

        return self.total_objective

    def reset_simulation(self):
        self.current_scenario_status = "episode_continues"
        self.iteration_counter = 0
        self.episode_counter = 0

    def iteration_update(self):
        self.total_objective_episodic_means.append(
            rc.mean(self.total_objectives_of_episodes)
        )

    def update_time_from_cache(self):
        self.time, self.episode_counter, self.iteration_counter = next(
            self.cached_timeline
        )

    def memorize(step_method):
        """Memorize the output of decorated method.

        It containes a ``cache`` field that in turn comprises ``keys`` and ``values``.
        The ``cache`` dictionary method `keys` returns a triple:

        - ``time``: the current time in an episode
        - ``episode_counter``: the current episode number
        - ``iteration_counter``: the current agent learning epoch

        If the scenario's triple ``(time, episode_counter, iteration_counter)`` is already contained in ``cache``, then the decorator returns a step method that simply reads from ``cache``.
        Otherwise, the scenario's simulator is called to do a step.
        """

        def step_with_memory(self):
            triple = (
                self.time,
                self.episode_counter,
                self.iteration_counter,
            )

            if triple in self.cache.keys():
                (
                    self.time,
                    self.episode_counter,
                    self.iteration_counter,
                    self.state,
                    self.action,
                    self.observation,
                    self.running_objective_value,
                    self.total_objective,
                    self.policy.model.weights,
                    self.critic.model.weights,
                    self.current_scenario_status,
                ) = self.cache[triple]

                self.update_time_from_cache()
            else:
                if self.howanim in ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK:
                    self.current_scenario_snapshot = [
                        self.time,
                        self.episode_counter,
                        self.iteration_counter,
                        self.state,
                        self.action,
                        self.observation,
                        self.running_objective_value,
                        self.total_objective,
                        self.policy.model.weights,
                        self.critic.model.weights,
                        self.current_scenario_status,
                    ]
                    self.cache[
                        (self.time, self.episode_counter, self.iteration_counter)
                    ] = self.current_scenario_snapshot

                self.current_scenario_status = step_method(self)

                if (
                    self.current_scenario_status == "simulation_ended"
                    and self.howanim
                    in ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK
                ):
                    keys = list(self.cache.keys())
                    current_scenario_status_idx = (
                        len(self.current_scenario_snapshot) - 1
                    )
                    for i, item in enumerate(self.cache.items()):
                        if i > self.speedup:
                            if (
                                item[1][current_scenario_status_idx]
                                != "episode_continues"
                            ):
                                key_i = keys[i]
                                for k in range(i - self.speedup + 1, i):
                                    key_k = keys[k]
                                    self.cache[key_k][
                                        current_scenario_status_idx
                                    ] = self.cache[key_i][current_scenario_status_idx]

                    # generator self.cached_timeline is needed for playback.
                    # self.cached_timeline skips frames depending on self.speedup using islice.
                    self.cached_timeline = islice(
                        cycle(iter(self.cache)), 0, None, self.speedup
                    )

            return self.current_scenario_status

        return step_with_memory

    # TODO: DOCSTRING
    @memorize
    def step(self):
        # sim_status = self.simulator.do_sim_step()
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
            ) = self.simulator.get_sim_step_data()

            self.delta_time = (
                self.time - self.time_old
                if self.time_old is not None and self.time is not None
                else 0
            )
            self.time_old = self.time

            # In future versions state vector being passed into controller should be obtained from an observer.
            self.action = self.controller.compute_action_sampled(
                self.time,
                self.state,
                self.observation,
            )
            self.simulator.receive_action(self.action)
            self.is_episode_ended = self.simulator.do_sim_step() == -1
            return "episode_continues"
        else:
            self.reset_episode()
            is_iteration_ended = self.episode_counter >= self.N_episodes

            if is_iteration_ended:
                self.iteration_update()
                self.reset_iteration()

                is_simulation_ended = self.iteration_counter >= self.N_iterations

                if is_simulation_ended:
                    self.reset_simulation()
                    return "simulation_ended"
                else:
                    return "iteration_ended"
            else:
                return "episode_ended"
