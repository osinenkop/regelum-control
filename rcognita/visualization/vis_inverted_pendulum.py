import numpy as np
from .animator import update_line, update_text, init_data_cursor, Dashboard, Animator
from ..__utilities import rc
import matplotlib.pyplot as plt


class InvPendulumTrackingDashboard(Dashboard):
    def __init__(self, time_start, rod_length, state_init, scenario):
        super().__init__()
        self.time_start = time_start
        self.rod_length = rod_length
        self.state_init = state_init
        self.scenario = scenario

    def init_dashboard(self):
        self.axes_rotating_pendulum = plt.gca()

        x_from = -self.rod_length - 0.2
        x_to = self.rod_length + 0.2
        y_from = x_from
        y_to = x_to
        self.axes_rotating_pendulum.autoscale(False)
        self.axes_rotating_pendulum.set_xlim(x_from, x_to)
        self.axes_rotating_pendulum.set_ylim(y_from, y_to)
        self.axes_rotating_pendulum.set_xlabel("x [m]")
        self.axes_rotating_pendulum.set_ylabel("y [m]")
        self.axes_rotating_pendulum.set_title(
            "Pause - space, q - quit, click - data cursor"
        )
        self.axes_rotating_pendulum.set_aspect("equal", adjustable="box")
        self.axes_rotating_pendulum.plot(
            [x_from, x_to], [0, 0], "k--", lw=0.75
        )  # Help line
        self.axes_rotating_pendulum.plot(
            [0, 0], [y_from, y_to], "k-", lw=0.75
        )  # Help line
        text_time = f"Time = {self.time_start:2.3f}"
        self.text_time_handle = self.axes_rotating_pendulum.text(
            0.05,
            0.95,
            text_time,
            horizontalalignment="left",
            verticalalignment="center",
            transform=self.axes_rotating_pendulum.transAxes,
        )
        self.axes_rotating_pendulum.format_coord = (
            lambda state, observation: "%2.2f, %2.2f" % (state, observation,)
        )

        xCoord0 = self.rod_length * rc.sin(self.state_init[0])
        yCoord0 = self.rod_length * rc.cos(self.state_init[0])

        self.scatter_sol = self.axes_rotating_pendulum.scatter(
            xCoord0, yCoord0, marker="o", s=400, c="b"
        )
        (self.line_rod,) = self.axes_rotating_pendulum.plot(
            [0, xCoord0], [0, yCoord0], "b", lw=1.5,
        )

        self.artists.append(self.text_time_handle)
        self.artists.append(self.line_rod)

    def perform_step_update(self):
        state_full = self.scenario.observation
        angle = state_full[0]

        xCoord = self.rod_length * rc.sin(angle)
        yCoord = self.rod_length * rc.cos(angle)

        text_time = f"Time = {self.scenario.time:2.3f}"
        update_text(self.text_time_handle, text_time)

        self.line_rod.set_xdata([0, xCoord])
        self.line_rod.set_ydata([0, yCoord])

        self.scatter_sol.remove()
        self.scatter_sol = self.axes_rotating_pendulum.scatter(
            xCoord, yCoord, marker="o", s=400, c="b"
        )


class EpisodicTrajectoryDashboard(Dashboard):
    def __init__(self, time_start, time_final, angle_0, scenario):
        super().__init__()
        self.time_start = time_start
        self.time_final = time_final
        self.angle_0 = angle_0
        self.N_episodes = scenario.N_episodes
        self.scenario = scenario

    def init_dashboard(self):
        self.axs_sol = plt.gca()
        self.axs_sol.plot(
            autoscale_on=False,
            xlim=(self.time_start, self.time_final),
            ylim=(-2 * np.pi, 2 * np.pi),
            xlabel="Time [s]",
        )

        self.episodic_line_handles = []
        for _ in range(self.N_episodes):
            (new_handle,) = self.axs_sol.plot([self.time_start], [0], "r--")
            self.episodic_line_handles.append(new_handle)

        self.artists.extend(self.episodic_line_handles)

        self.axs_sol.plot(
            [self.time_start, self.time_final], [0, 0], "k--", lw=0.75
        )  # Help line
        (self.line_angle,) = self.axs_sol.plot(
            self.time_start, self.angle_0, "r", lw=0.5, label=r"$\angle$ [rad]"
        )
        init_data_cursor(self.line_angle)

        self.axs_sol.legend(fancybox=True, loc="upper right")
        self.axs_sol.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )

        self.artists.append(self.line_angle)

    def perform_step_update(self):
        state_full = self.scenario.observation
        angle = state_full[0]
        time = self.scenario.time
        update_line(self.line_angle, time, angle)

    def perform_episodic_update(self):
        # self.perform_step_update()
        x_data = self.line_angle.get_xdata()
        y_data = self.line_angle.get_ydata()
        handle = self.episodic_line_handles[self.scenario.episode_counter - 1]
        handle.set_xdata(x_data[:-1])
        handle.set_ydata(y_data[:-1])
        self.line_angle.set_xdata([self.time_start])
        self.line_angle.set_ydata([self.scenario.state_init[0]])

    def perform_iterative_update(self):
        self.perform_episodic_update()
        for handle in self.episodic_line_handles:
            handle.set_xdata([self.time_start])
            handle.set_ydata([self.scenario.state_init[0]])


class MeanEpisodicOutcomesDashboard(Dashboard):
    def __init__(self, time_start, time_final, scenario):
        super().__init__()
        self.time_start = time_start
        self.time_final = time_final
        self.scenario = scenario

    def init_dashboard(self):
        self.axes_cost = plt.gca()

        self.axes_cost.set_xlim(self.time_start, self.time_final)
        self.axes_cost.set_ylim(-1e5, 0)
        self.axes_cost.set_yscale("symlog")
        self.axes_cost.set_ylabel("Outcome")
        self.axes_cost.set_xlabel("Iteration number")
        self.axes_cost.autoscale(False)

        (self.line_outcome_episodic_mean,) = self.axes_cost.plot(
            [], [], "r-", lw=0.5, label="Iteration mean outcome"
        )
        self.axes_cost.legend(fancybox=True, loc="upper right")
        self.axes_cost.grid()

        init_data_cursor(self.line_outcome_episodic_mean)
        self.artists.append(self.line_outcome_episodic_mean)

    def perform_iterative_update(self):
        update_line(
            self.line_outcome_episodic_mean,
            self.scenario.iteration_counter,
            self.scenario.outcome_episodic_means[-1],
        )


class WeightsEpisodicDashboard(Dashboard):
    def __init__(self, N_iterations, weights_init, scenario):
        super().__init__()
        self.N_iterations = N_iterations
        self.weights_init = weights_init
        self.scenario = scenario

    def init_dashboard(self):
        self.axs_action_params = plt.gca()

        self.axs_action_params.set_xlim(0, self.N_iterations)
        self.axs_action_params.set_xlabel("Iteration number")

        self.axs_action_params.set_ylim(0, 100)

        self.policy_line_handles_pack = [
            self.axs_action_params.plot(
                [0], [self.scenario.actor.model.weights_init[i]], label=f"w_{i+1}",
            )[0]
            for i in range(len(self.weights_init))
        ]
        self.artists.extend(self.policy_line_handles_pack)
        plt.legend()

    def perform_iterative_update(self):
        for i, handle in enumerate(self.policy_line_handles_pack):
            update_line(
                handle,
                self.scenario.iteration_counter,
                self.scenario.actor.model.weights[i],
            )


class AnimatorInvertedPendulum(Animator):
    def __init__(self, scenario, subplot_grid_size=[2, 2]):
        super().__init__(subplot_grid_size=subplot_grid_size)

        self.__dict__.update(scenario.__dict__)

        self.scenario = scenario

        self.system = self.scenario.simulator.system

        self.sampling_time = self.scenario.controller.sampling_time
        # Unpack entities

        # Store some parameters for later use
        self.time_old = 0
        self.outcome = 0

        self.angle_0 = self.state_init[0]
        self.rod_length = self.system.pars[2]

        ########### SUBPLOT 1  --------- PENDULUM TRACKING ################
        pendulum_tracking_dashboard = InvPendulumTrackingDashboard(
            self.time_start, self.rod_length, self.state_init, self.scenario
        )
        ########### SUBPLOT 2  --------- STEP-BY-STEP-SOLUTION ################
        episodic_trajectory_dashboard = EpisodicTrajectoryDashboard(
            self.time_start, self.time_final, self.angle_0, self.scenario
        )
        ########### SUBPLOT 3  --------- Episode mean ################
        mean_outcomes_dashboard = MeanEpisodicOutcomesDashboard(
            self.time_start, self.time_final, self.scenario
        )

        ########### SUBPLOT 4  --------- POLICY PARAMETERS ################
        weights_episodic_dashboard = WeightsEpisodicDashboard(
            self.scenario.N_iterations,
            self.scenario.actor.model.weights_init,
            self.scenario,
        )
        ###################################################################

        self.collect_dashboards(
            pendulum_tracking_dashboard,
            episodic_trajectory_dashboard,
            mean_outcomes_dashboard,
            weights_episodic_dashboard,
        )
        self.run_curr = 1

    def reset(self):
        self.current_step = 0

        self.line_angle.set_xdata([])
        self.line_angle.set_ydata([])
        for handle in self.episodic_line_handles:
            handle.set_xdata([self.time_start])
            handle.set_ydata([self.state_init[0]])

        for i, handle in enumerate(self.policy_line_handles_pack):
            handle.set_xdata(self.iters[self.current_step])
            handle.set_ydata(self.weights[self.current_step][i])

        self.line_outcome_episodic_mean.set_xdata([])
        self.line_outcome_episodic_mean.set_ydata([])
        self.episodic_outcomes = []
        self.iteration_counter = 0
        self.episode_counter = 0
        self.scenario.logger.reset()
