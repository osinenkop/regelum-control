"""
This module contains various simulation scenarios.
For instance, an online scenario is when the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

"""


from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from itertools import islice, cycle
import numpy as np
from typing import Optional
from unittest.mock import Mock, MagicMock

import rcognita.base
from .__utilities import rc
from .optimizers import TorchOptimizer
from .actors import Actor
from .critics import Critic, CriticTrivial
from .simulator import Simulator
from .controllers import Controller
from .objectives import RunningObjective
from .callbacks import apply_callbacks
from . import ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK

try:
    import torch
except ImportError:
    torch = MagicMock()


class Scenario(rcognita.base.RcognitaBase, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def step(self):
        pass


class TabularScenarioVI(Scenario):
    """
    Tabular scenario for the use with tabular agents.
    Each iteration entails processing of the whole observation (or state) and action spaces, correponds to a signle update of the agent.
    Implements a scenario for value iteration (VI) update.

    """

    def __init__(
        self, actor: Actor, critic: Critic, N_iterations: int, is_playback=None
    ):
        self.actor = actor
        self.critic = critic
        self.N_iterations = N_iterations

    def run(self):
        for i in range(self.N_iterations):
            self.step()

    def step(self):
        self.actor.update()
        self.critic.update()


class OnlineScenario(Scenario):
    cache = dict()

    def __init__(
        self,
        simulator: Simulator,
        controller: Controller,
        running_objective: Optional[RunningObjective] = None,
        is_log: bool = False,
        howanim: str = None,
        state_init: np.ndarray = None,
        action_init: np.ndarray = None,
        time_start: float = 0.0,
        observation_target: list = [],
        observation_components_naming=[],
        N_episodes=1,
        N_iterations=1,
        speedup=1,
    ):

        self.cache.clear()
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.weights_historical = []
        self.simulator = simulator

        self.system = Mock() if not hasattr(simulator, "system") else simulator.system

        self.controller = controller
        self.actor = (
            MagicMock() if not hasattr(controller, "actor") else controller.actor
        )
        self.critic = (
            CriticTrivial(running_objective)
            if not hasattr(controller, "critic")
            else controller.critic
        )

        self.running_objective = (
            (lambda observation, action: 0)
            if running_objective is None
            else running_objective
        )
        if observation_target != []:
            self.running_objective.observation_target = rc.array(observation_target)
            if hasattr(self.actor, "running_objective"):
                self.actor.running_objective.observation_target = rc.array(
                    observation_target
                )
            if hasattr(self.critic, "running_objective"):
                self.critic.running_objective.observation_target = rc.array(
                    observation_target
                )
        self.time_start = time_start
        self.time_final = self.simulator.time_final
        self.is_log = is_log
        self.howanim = howanim
        self.is_playback = (
            self.howanim in ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK
        )
        self.state_init = state_init
        self.state_full = state_init
        self.action_init = action_init
        self.action = self.action_init
        self.observation = self.system.out(self.state_init)
        self.observation_target = (
            np.zeros_like(self.observation)
            if observation_target is None or observation_target == []
            else rc.array(observation_target)
        )
        self.running_objective_value = self.running_objective(
            self.observation, self.action
        )
        self.total_objective = 0
        self.time = 0
        self.time_old = 0
        self.delta_time = 0
        self.observation_components_naming = observation_components_naming

        self.recent_total_objectives_of_episodes = []
        self.total_objectives_of_episodes = []
        self.total_objective_episodic_means = []
        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.current_scenario_status = "episode_continues"
        self.speedup = speedup

    def set_speedup(self, speedup):
        self.speedup = speedup
        self.cached_timeline = islice(cycle(iter(self.cache)), 0, None, self.speedup)

    def get_speeduped_cache_len(self):
        return len(range(0, len(self.cache), self.speedup))

    def get_cache_len(self):
        return len(self.cache)

    def update_total_objective(self, observation, action, delta):

        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``outcome`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize objective instead).

        """

        self.total_objective += self.running_objective(observation, action) * delta

    @apply_callbacks()
    def reload_pipeline(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.recent_total_objective = self.total_objective
        self.total_objective = 0
        self.action = self.action_init
        self.system.reset()
        self.actor.reset()
        self.critic.reset()
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        self.simulator.reset()
        self.observation = self.system.out(self.state_init, time=0)
        self.sim_status = 0
        return self.recent_total_objective

    def run(self):
        ### We use values `iteration_counter` and `episode_counter` only for debug purposes
        for iteration_counter in range(self.N_iterations):
            for episode_counter in range(self.N_episodes):
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

    def reset_episode(self):
        self.total_objectives_of_episodes.append(self.total_objective)
        self.episode_counter += 1
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
        """
        This is a decorator for a simulator step method.
        It containes a ``cache`` field that in turn comprises of ``keys`` and ``values``.
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
                    self.state_full,
                    self.action,
                    self.observation,
                    self.running_objective_value,
                    self.total_objective,
                    self.actor.model.weights,
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
                        self.state_full,
                        self.action,
                        self.observation,
                        self.running_objective_value,
                        self.total_objective,
                        self.actor.model.weights,
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

    @apply_callbacks()
    def pre_step(self):
        self.running_objective_value = self.running_objective(
            self.observation, self.action
        )
        self.update_total_objective(self.observation, self.action, self.delta_time)

        return (
            np.around(self.running_objective_value, decimals=2),
            self.observation.round(decimals=2),
            self.action.round(2),
            self.total_objective,
        )

    @apply_callbacks()
    def post_step(self):
        self.running_objective_value = self.running_objective(
            self.observation, self.action
        )
        self.update_total_objective(self.observation, self.action, self.delta_time)

        return (
            np.around(self.running_objective_value, decimals=2),
            self.observation.round(decimals=2),
            self.action.round(2),
            self.total_objective,
        )

    @memorize
    def step(self):
        self.pre_step()
        sim_status = self.simulator.do_sim_step()
        is_episode_ended = sim_status == -1

        if not is_episode_ended:
            (
                self.time,
                self.state,
                self.observation,
                self.state_full,
            ) = self.simulator.get_sim_step_data()

            self.delta_time = self.time - self.time_old
            self.time_old = self.time

            # In future versions state vector being passed into controller should be obtained from an observer.
            self.action = self.controller.compute_action_sampled(
                self.time,
                self.state,
                self.observation,
                observation_target=self.observation_target,
            )
            self.system.receive_action(self.action)
            self.post_step()

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


class MonteCarloScenario(OnlineScenario):
    def step(self):
        episode_status = super().step()
        if self.controller.is_time_for_new_sample:
            self.controller.episode_data_buffer.add_step_data(
                observation=self.observation,
                action=self.action,
                running_objective=self.running_objective_value,
                current_total_objective=self.total_objective,
                episode_id=self.episode_counter,
                is_step_done=episode_status != "episode_continues",
            )

        return episode_status

    def reset_iteration(self):
        self.controller.episode_data_buffer.set_total_objectives_of_episodes(
            self.total_objectives_of_episodes
        )

        if self.current_scenario_status != "simulation_ended":
            self.actor.optimize_weights_after_iteration(
                self.controller.episode_data_buffer
            )
            self.controller.episode_data_buffer.nullify_buffer()
        super().reset_episode()
        super().reset_iteration()
