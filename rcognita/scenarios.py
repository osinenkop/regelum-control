"""
This module contains various simulation scenarios.
For instance, an online scenario is when the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

"""

from re import S
from rcognita.utilities import rc
from rcognita.optimizers import TorchOptimizer
from abc import ABC, abstractmethod
from copy import deepcopy
import matplotlib.pyplot as plt
import sys


class TabularScenarioBase:
    """
    A tabular scenario blueprint.

    """

    def __init__(self, actor, critic, N_iterations):
        self.actor = actor
        self.critic = critic
        self.N_iterations = N_iterations

    def run(self):
        for i in range(self.N_iterations):
            self.iterate()

    @abstractmethod
    def iterate(self):
        pass


class TabularScenarioVI(TabularScenarioBase):
    """
    Tabular scenario for the use with tabular agents.
    Each iteration entails processing of the whole observation (or state) and action spaces, correponds to a signle update of the agent.
    Implements a scenario for value iteration (VI) update.

    """

    def iterate(self):
        self.actor.update()
        self.critic.update()


class TabularScenarioPI(TabularScenarioBase):
    """
    Tabular scenario for the use with tabular agents.
    Each iteration entails processing of the whole observation (or state) and action spaces, correponds to a signle update of the agent.
    Implements a scenario for policy iteration (PI) update.
    """

    def iterate(self):
        self.critic.update()
        self.actor.update()


class OnlineScenario:
    """
    Online scenario: the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

    """

    def __init__(
        self,
        system,
        simulator,
        controller,
        actor,
        critic,
        logger,
        datafiles,
        time_final,
        running_objective,
        no_print=False,
        is_log=False,
        is_playback=False,
        state_init=None,
        action_init=None,
    ):
        self.system = system
        self.simulator = simulator
        self.controller = controller
        self.actor = actor
        self.critic = critic
        self.logger = logger
        self.running_objective = running_objective

        self.time_final = time_final
        self.datafile = datafiles[0]
        self.no_print = no_print
        self.is_log = is_log
        self.is_playback = is_playback
        self.state_init = state_init
        self.action_init = action_init

        self.trajectory = []
        self.outcome = 0
        self.time_old = 0
        self.delta_time = 0
        if self.is_playback:
            self.episodic_playback_table = []

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
        if self.is_log:
            self.logger.log_data_row(
                self.datafile,
                self.time,
                self.state_full,
                self.action,
                self.running_objective_value,
                self.outcome,
            )

        if self.is_playback:
            self.episodic_playback_table.append(
                [
                    self.time,
                    *self.state_full,
                    *self.action,
                    self.running_objective_value,
                    self.outcome,
                ]
            )

    def run(self):
        while True:
            is_episode_ended = self.step() == 1

            if is_episode_ended:
                print("Episode ended successfully.")
                break

    def step(self):
        sim_status = self.simulator.do_sim_step()
        is_episode_ended = sim_status == -1

        if is_episode_ended:
            return -1

        (
            self.time,
            _,
            self.observation,
            self.state_full,
        ) = self.simulator.get_sim_step_data()
        self.trajectory.append(rc.concatenate((self.state_full, self.time), axis=None))

        self.delta_time = self.time - self.time_old
        self.time_old = self.time

        self.action = self.controller.compute_action_sampled(
            self.time, self.observation
        )
        self.system.receive_action(self.action)

        self.perform_post_step_operations()

    def update_outcome(self, observation, action, delta):

        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``outcome`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize objective instead).

        """

        self.outcome += self.running_objective(observation, action) * delta


class EpisodicScenarioBase(OnlineScenario):
    def __init__(
        self, N_episodes, N_iterations, *args, **kwargs,
    ):
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
        if self.is_playback:
            self.episode_tables = []

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

        if self.is_playback:
            if len(self.episode_tables) > 1:
                self.episode_tables = rc.vstack(self.episode_tables)
            else:
                self.episode_tables = rc.array(self.episode_tables[0])

    def get_table_from_last_episode(self):
        return rc.array(
            [
                rc.array(
                    [
                        self.iteration_counter,
                        self.episode_counter,
                        *x,
                        *self.actor.model.weights,
                    ]
                )
                for x in self.episodic_playback_table
            ]
        )

    def reset_iteration(self):
        self.episode_counter = 0
        self.iteration_counter += 1
        self.outcomes_of_episodes = []
        self.episode_REINFORCE_objective_gradients = []

    def reset_episode(self):
        if self.is_playback:
            new_table = self.get_table_from_last_episode()
            self.episode_tables.append(new_table)
            self.episodic_playback_table = []

        self.outcomes_of_episodes.append(self.critic.outcome)
        self.episode_counter += 1

    def iteration_update(self):
        self.outcome_episodic_means.append(rc.mean(self.outcomes_of_episodes))

    def step(self):
        sim_status = super().step()

        is_episode_ended = sim_status == -1

        if is_episode_ended:
            self.reset_episode()

            is_iteration_ended = self.episode_counter >= self.N_episodes

            if is_iteration_ended:
                self.iteration_update()
                self.reset_iteration()

                is_simulation_ended = self.iteration_counter >= self.N_iterations

                if is_simulation_ended:
                    return "simulation_ended"
                else:
                    return "iteration_ended"
            else:
                return "episode_ended"


class EpisodicScenario(EpisodicScenarioBase):
    def __init__(
        self,
        *args,
        learning_rate=0.001,
        is_fixed_actor_weights=False,
        is_plot_critic=True,
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


class EpisodicScenarioAsyncAC(EpisodicScenario):
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


class EpisodicScenarioCriticLearn(EpisodicScenario):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import numpy as np

        angle_inits = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, self.N_iterations)
        angular_velocity_inits = np.random.uniform(
            -np.pi / 2.0, np.pi / 2.0, self.N_iterations
        )

        w1s = np.random.uniform(0, 15, self.N_iterations)
        w2s = np.random.uniform(0, 15, self.N_iterations)
        w3s = np.random.uniform(0, 15, self.N_iterations)

        self.state_inits = np.vstack((angle_inits, angular_velocity_inits)).T
        self.actor_model_weights = np.vstack((w1s, w2s, w3s)).T

        self.action_inits = np.random.uniform(-25.0, 25.0, self.N_iterations)
        self.critic_loss_values = []

    def init_conditions_update(self):
        self.simulator.state_full_init = self.state_init = self.state_inits[
            self.iteration_counter, :
        ]
        self.action_init = self.action_inits[self.iteration_counter]
        self.actor.model.weights = self.actor_model_weights[self.iteration_counter, :]

    def reload_pipeline(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.outcome = 0
        self.init_conditions_update()
        self.action = self.action_init
        self.system.reset()
        self.actor.reset()
        self.critic.reset()
        self.controller.reset(time_start=0)
        self.simulator.reset()
        self.observation = self.system.out(self.state_init, time=0)
        self.sim_status = 0

    def run(self):
        self.step_counter = 0
        self.one_episode_steps_numbers = [0]
        skipped_steps = 43
        for _ in range(self.N_iterations):
            for _ in range(self.N_episodes):
                while self.sim_status not in [
                    "episode_ended",
                    "simulation_ended",
                    "iteration_ended",
                ]:
                    self.sim_status = self.step()
                    self.step_counter += 1
                    if self.step_counter > skipped_steps:
                        self.critic_loss_values.append(self.critic.current_critic_loss)

                self.one_episode_steps_numbers.append(
                    self.one_episode_steps_numbers[-1]
                    + self.step_counter
                    - skipped_steps
                )
                self.step_counter = 0
                if self.sim_status != "simulation_ended":
                    self.reload_pipeline()
        if self.is_playback:
            if len(self.episode_tables) > 1:
                self.episode_tables = rc.vstack(self.episode_tables)
            else:
                self.episode_tables = rc.array(self.episode_tables[0])

        self.plot_critic_learn_results()

    def plot_critic_learn_results(self):
        figure = plt.figure(figsize=(9, 9))
        ax_critic = figure.add_subplot(111)
        ax_critic.plot(self.critic_loss_values, label="TD")
        [ax_critic.axvline(i, c="r") for i in self.one_episode_steps_numbers]
        plt.legend()
        plt.savefig(
            f"./critic/{self.N_iterations}-iters_{self.time_final}-fintime_{self.critic.data_buffer_size}-dbsize",
            format="png",
        )
        plt.show()
