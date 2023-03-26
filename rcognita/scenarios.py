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
    """
    Online scenario: the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

    """

    def __init__(
        self,
        simulator: Simulator,
        controller: Controller,
        running_objective: Optional[RunningObjective] = None,
        no_print: bool = False,
        is_log: bool = False,
        howanim: str = None,
        state_init: np.ndarray = None,
        action_init: np.ndarray = None,
        time_start: float = 0.0,
        observation_target=[],
        observation_components_naming=[],
    ):
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
        self.no_print = no_print
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

        self.trajectory = []
        self.outcome = 0
        self.time = 0
        self.time_old = 0
        self.delta_time = 0
        self.observation_components_naming = observation_components_naming

    @apply_callbacks()
    def pre_step(self):
        self.running_objective_value = self.running_objective(
            self.observation, self.action
        )
        self.update_outcome(self.observation, self.action, self.delta_time)

        return (
            np.around(self.running_objective_value, decimals=2),
            self.observation.round(decimals=2),
            self.action.round(2),
            self.outcome,
        )

    @apply_callbacks()
    def post_step(self):
        self.running_objective_value = self.running_objective(
            self.observation, self.action
        )
        self.update_outcome(self.observation, self.action, self.delta_time)

        return (
            np.around(self.running_objective_value, decimals=2),
            self.observation.round(decimals=2),
            self.action.round(2),
            self.outcome,
        )

    def run(self):

        while self.step():
            pass
        print("Episode ended successfully.")

    def step(self):
        self.pre_step()
        sim_status = self.simulator.do_sim_step()
        is_episode_ended = sim_status == -1

        if is_episode_ended:
            return False
        else:

            (
                self.time,
                self.state,
                self.observation,
                self.state_full,
            ) = self.simulator.get_sim_step_data()

            self.trajectory.append(
                rc.concatenate((self.state_full, self.time), axis=None)
            )

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

            return True

    def update_outcome(self, observation, action, delta):

        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``outcome`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize objective instead).

        """

        self.outcome += self.running_objective(observation, action) * delta


class EpisodicScenario(OnlineScenario):
    cache = dict()

    def __init__(
        self,
        N_episodes,
        N_iterations,
        *args,
        speedup=1,
        **kwargs,
    ):
        self.cache.clear()
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.weights_historical = []
        super().__init__(*args, **kwargs)
        self.outcomes_of_episodes = []
        self.outcome_episodic_means = []
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

    @apply_callbacks()
    def reload_pipeline(self):
        return self.reload_pipeline_no_callbacks()

    def reload_pipeline_no_callbacks(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        outcome = self.outcome
        self.outcome = 0
        self.action = self.action_init
        self.system.reset()
        self.actor.reset()
        self.critic.reset()
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        self.simulator.reset()
        self.observation = self.system.out(self.state_init, time=0)
        self.sim_status = 0
        return outcome

    def run(self):
        for _ in range(self.N_iterations):
            for _ in range(self.N_episodes):
                while self.sim_status not in [
                    "episode_ended",
                    "simulation_ended",
                    "iteration_ended",
                ]:
                    self.sim_status = self.step()

                self.reload_pipeline()
        outcome = self.outcome
        self.reset_episode()
        self.reset_iteration()
        self.reset_simulation()
        return outcome

    def reset_iteration(self):
        self.episode_counter = 0
        self.iteration_counter += 1
        self.outcomes_of_episodes = []

    def reset_episode(self):
        self.outcomes_of_episodes.append(self.critic.outcome)
        self.episode_counter += 1
        return self.critic.outcome

    def reset_simulation(self):
        self.current_scenario_status = "episode_continues"
        self.iteration_counter = 0
        self.episode_counter = 0

    def iteration_update(self):
        self.outcome_episodic_means.append(rc.mean(self.outcomes_of_episodes))

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
                    self.outcome,
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
                        self.outcome,
                        self.actor.model.weights,
                        self.critic.model.weights,
                        self.current_scenario_status,
                    ]
                    self.cache[
                        (self.time, self.episode_counter, self.iteration_counter)
                    ] = self.current_scenario_snapshot

                self.current_scenario_status = step_method(self)

                if self.current_scenario_status == "simulation_ended":
                    keys = list(self.cache.keys())
                    for i, item in enumerate(self.cache.items()):
                        if i > self.speedup:
                            if item[1][10] != "episode_continues":
                                key_i = keys[i]
                                for k in range(i - self.speedup + 1, i):
                                    key_k = keys[k]
                                    self.cache[key_k][10] = self.cache[key_i][10]

                    # generator self.cached_timeline is needed for playback.
                    # self.cached_timeline skips frames depending on self.speedup using islice.
                    self.cached_timeline = islice(
                        cycle(iter(self.cache)), 0, None, self.speedup
                    )

            return self.current_scenario_status

        return step_with_memory

    @memorize
    def step(self):
        episode_not_ended = super().step()

        if not episode_not_ended:
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
        else:
            return "episode_continues"


class EpisodicScenarioREINFORCE(EpisodicScenario):
    def __init__(
        self,
        *args,
        learning_rate=0.001,
        is_fixed_actor_weights=False,
        is_plot_critic=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.is_fixed_actor_weights = is_fixed_actor_weights
        self.is_plot_critic = is_plot_critic

    def reset_episode(self):
        super().reset_episode()
        self.store_REINFORCE_objective_gradient()

    def store_REINFORCE_objective_gradient(self):
        episode_REINFORCE_objective_gradient = self.critic.outcome * sum(
            self.actor.gradients
        )
        self.episode_REINFORCE_objective_gradients.append(
            episode_REINFORCE_objective_gradient
        )

    def get_mean_REINFORCE_gradient(self):
        return sum(self.episode_REINFORCE_objective_gradients) / len(
            self.episode_REINFORCE_objective_gradients
        )

    def iteration_update(self):
        super().iteration_update()
        mean_REINFORCE_gradient = self.get_mean_REINFORCE_gradient()

        if self.is_fixed_actor_weights == False:
            self.actor.update_weights_by_gradient(
                mean_REINFORCE_gradient, self.learning_rate
            )

    def run(self):
        super().run()

        if self.is_plot_critic:
            self.plot_critic()

    def plot_critic(self):
        self.fig_critic = plt.figure(figsize=(10, 10))
        ax_TD_means = self.fig_critic.add_subplot(111)
        ax_TD_means.plot(
            self.square_TD_means,
            label="square TD means\nby episode",
            c="r",
            scaley="symlog",
        )
        plt.legend()
        plt.savefig(
            f"./critic_plots/{self.N_iterations}-iters_{self.N_episodes}-episodes_{self.time_final}-fintime",
            format="png",
        )

    def get_mean(self, array):
        return sum(array) / len(array)


class ReplayBuffer:
    def __init__(self, dim_observation, dim_action, max_size=int(1e6)):
        self.max_size = max_size
        self.n_columns_filled = 0
        self.state = np.zeros((dim_observation, max_size))
        self.action = np.zeros((dim_action, max_size))
        self.reward = np.zeros((1, max_size))
        self.not_done = np.zeros((1, max_size))

    def add(self, state, action, reward, done):
        self.state = rc.push_vec(self.state, state)
        self.action = rc.push_vec(self.action, action)
        self.reward = rc.push_vec(self.reward, reward)
        self.not_done = rc.push_vec(self.not_done, done)
        self.n_columns_filled = min(self.n_columns_filled + 1, self.max_size)

    def sample(self, batch_size=1, random_idxs=False):
        if random_idxs:
            ind = np.random.randint(0, self.n_columns_filled, size=batch_size)
        else:
            ind = np.arange(-min(self.n_columns_filled, batch_size), 0)
        return (
            torch.DoubleTensor(self.state[:, ind]).T,
            torch.DoubleTensor(self.action[:, ind]).T,
            torch.DoubleTensor(self.reward[:, ind]).T,
            torch.DoubleTensor(self.not_done[:, ind]).T,
        )

    def nullify_buffer(self):
        self.n_columns_filled = 0
        self.state *= 0
        self.action *= 0
        self.reward *= 0
        self.not_done *= 0


class EpisodicScenarioMultirun(EpisodicScenario):
    def __init__(self, repeat_num: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat_num = repeat_num


class EpisodicScenarioTorchREINFORCE(EpisodicScenarioMultirun):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim_observation = len(self.state_init)
        self.dim_action = len(self.action_init)
        self.replay_buffer = ReplayBuffer(
            dim_observation=self.dim_observation,
            dim_action=self.dim_action,
            max_size=int(
                10 / self.controller.sampling_time * self.N_episodes * self.time_final
            ),
        )
        self.mean_q_values = np.zeros(self.N_episodes)

    def step(self):
        episode_status = super().step()
        self.replay_buffer.add(
            self.observation,
            self.action,
            self.running_objective_value,
            episode_status != "episode_continues",
        )
        return episode_status

    def reset_episode(self):
        observations, _, _, _ = self.replay_buffer.sample(
            int(
                self.simulator.time_final
                * self.N_episodes
                / self.controller.sampling_time
            )
        )
        self.actor.optimize_weights_episodic(observations)
        self.replay_buffer.nullify_buffer()
        super().reset_episode()
