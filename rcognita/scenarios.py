"""
This module contains various simulation scenarios.
For instance, an online scenario is when the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

"""


from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from itertools import islice
import numpy as np
from typing import Optional
from unittest.mock import Mock, MagicMock


from .__utilities import rc
from .optimizers import TorchOptimizer
from .actors import Actor
from .critics import Critic
from .simulator import Simulator
from .controllers import Controller
from .objectives import RunningObjective


class Scenario(ABC):
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
        is_playback: bool = False,
        state_init: np.ndarray = None,
        action_init=None,
        time_start: float = 0.0,
    ):

        self.simulator = simulator

        self.system = Mock() if not hasattr(simulator, "system") else simulator.system

        self.controller = controller
        self.actor = (
            MagicMock() if not hasattr(controller, "actor") else controller.actor
        )
        self.critic = (
            MagicMock() if not hasattr(controller, "critic") else controller.critic
        )

        self.running_objective = (
            (lambda observation, action: 0)
            if running_objective is None
            else running_objective
        )

        self.time_start = time_start
        self.time_final = self.simulator.time_final
        self.no_print = no_print
        self.is_log = is_log
        self.is_playback = is_playback
        self.state_init = state_init
        self.state_full = state_init
        self.action_init = action_init
        self.action = self.action_init
        self.observation = self.system.out(self.state_full)
        self.running_objective_value = self.running_objective(
            self.observation, self.action
        )

        self.trajectory = []
        self.outcome = 0
        self.time = 0
        self.time_old = 0
        self.delta_time = 0

    def perform_post_step_operations(self):
        self.running_objective_value = self.running_objective(
            self.observation, self.action
        )
        self.update_outcome(self.observation, self.action, self.delta_time)

        if not self.no_print:
            self.logger.print_sim_step(
                self.time,
                self.state_full,
                self.action,
                self.running_objective_value,
                self.outcome,
            )

    def run(self):

        while self.step():
            pass
        print("Episode ended successfully.")

    def step(self):
        sim_status = self.simulator.do_sim_step()
        is_episode_ended = sim_status == -1

        if is_episode_ended:
            return False
        else:

            (
                self.time,
                _,
                self.observation,
                self.state_full,
            ) = self.simulator.get_sim_step_data()

            self.trajectory.append(
                rc.concatenate((self.state_full, self.time), axis=None)
            )

            self.delta_time = self.time - self.time_old
            self.time_old = self.time

            self.action = self.controller.compute_action_sampled(
                self.time, self.observation
            )
            self.system.receive_action(self.action)

            self.perform_post_step_operations()

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
        self, N_episodes, N_iterations, *args, speedup=1, **kwargs,
    ):
        self.cache.clear()
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.episode_REINFORCE_objective_gradients = []
        self.weights_historical = []
        super().__init__(*args, **kwargs)
        self.weights_historical.append(self.actor.model.weights[0])
        self.outcomes_of_episodes = []
        self.outcome_episodic_means = []
        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.current_scenario_status = "episode_continues"
        self.speedup = speedup

    def reload_pipeline(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.outcome = 0
        self.action = self.action_init
        self.system.reset()
        self.actor.reset()
        self.critic.reset()
        self.controller.reset(time_start=0)
        self.simulator.reset()
        self.observation = self.system.out(self.state_init, time=0)
        self.sim_status = 0

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

        self.reset_episode()
        self.reset_iteration()
        self.reset_simulation()

    def reset_iteration(self):
        self.episode_counter = 0
        self.iteration_counter += 1
        self.outcomes_of_episodes = []
        self.episode_REINFORCE_objective_gradients = []

    def reset_episode(self):
        self.outcomes_of_episodes.append(self.critic.outcome)
        self.episode_counter += 1

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
                if self.is_playback:
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
                    for i, item in enumerate(self.cache.items()):
                        if i > self.speedup:
                            if item[1][10] != "episode_continues":
                                key_i = list(self.cache.keys())[i]
                                for k in range(i - self.speedup + 1, i):
                                    key_k = list(self.cache.keys())[k]
                                    self.cache[key_k][10] = self.cache[key_i][10]

                    self.cached_timeline = islice(
                        iter(self.cache), 0, None, self.speedup
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


class EpisodicScenarioAsyncAC(EpisodicScenarioREINFORCE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_optimizer = TorchOptimizer({"lr": 0.01})
        self.squared_TD_sums_of_episodes = []
        self.square_TD_means = []

    def store_REINFORCE_objective_gradient(self):
        episode_REINFORCE_objective_gradient = sum(self.actor.gradients)
        self.episode_REINFORCE_objective_gradients.append(
            episode_REINFORCE_objective_gradient
        )

    def reset_episode(self):
        self.squared_TD_sums_of_episodes.append(self.critic.objective())
        super().reset_episode()

    def iteration_update(self):
        mean_sum_of_squared_TD = self.get_mean(self.squared_TD_sums_of_episodes)
        self.square_TD_means.append(mean_sum_of_squared_TD.detach().numpy())

        self.critic_optimizer.optimize(
            objective=self.get_mean,
            model=self.critic.model,
            model_input=self.squared_TD_sums_of_episodes,
        )

        super().iteration_update()

    def reset_iteration(self):
        self.squared_TD_sums_of_episodes = []
        super().reset_iteration()
