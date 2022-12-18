"""
This module contains the logger interface along with concrete realizations for each separate system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from tabulate import tabulate

import csv
import numpy as np


class Logger:
    """
    Interface class for data loggers.
    Concrete loggers, associated with concrete system-controller setups, are should be built upon this class.
    To design a concrete logger: inherit this class, override:
        | :func:`~loggers.Logger.print_sim_step` :
        | print a row of data of a single simulation step, typically into the console (required).
        | :func:`~loggers.Logger.log_data_row` :
        | same as above, but write to a file (required).

    """

    def __init__(
        self,
        state_components_strings,
        action_components_strings,
        row_format_list=None,
        N_episodes=1,
        N_iterations=1,
    ):
        self.state_components_strings = state_components_strings
        self.action_components_strings = action_components_strings
        self.N_episodes = N_episodes
        self.episodes_passed = 0
        self.N_iterations = N_iterations
        self.iterations_passed = 0
        self.is_next_episode = False
        self.time_old = 0

        is_many_episodes = int(self.N_episodes == 1)
        is_many_iterations = int(self.N_iterations == 1)

        self.row_header = [
            "iterations_passed",
            "episodes_passed",
            "t [s]",
            *self.state_components_strings,
            *self.action_components_strings,
            "running_objective",
            "outcome",
        ][is_many_episodes + is_many_iterations :]
        if row_format_list is None:
            self.row_format = tuple(["8.3f" for _ in self.row_header])
        else:
            self.row_format = row_format_list

    def print_sim_step(self, time, state_full, action, running_objective, outcome):
        self.is_next_episode = time < self.time_old

        if self.is_next_episode:
            self.episodes_passed += 1
            if self.N_episodes == self.episodes_passed:
                self.iterations_passed += 1
                self.episodes_passed = 0

        self.time_old = time

        is_many_episodes = int(self.N_episodes == 1)
        is_many_iterations = int(self.N_iterations == 1)

        row_data = [
            self.iterations_passed,
            self.episodes_passed,
            time,
            *np.array(state_full),
            *np.array(action),
            running_objective,
            outcome,
        ][is_many_episodes + is_many_iterations :]

        table = tabulate(
            [self.row_header, row_data],
            floatfmt=self.row_format,
            headers="firstrow",
            tablefmt="grid",
        )

        print(table)

    def log_data_row(
        self, datafile, time, state_full, action, running_objective, outcome
    ):
        if self.is_next_episode:
            self.episodes_passed += 1
            if self.N_episodes == self.episodes_passed:
                self.iterations_passed += 1
                self.episodes_passed = 0

        self.time_old = time

        with open(datafile, "a", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [
                    self.iterations_passed,
                    self.episodes_passed,
                    time,
                    *state_full,
                    *action,
                    running_objective,
                    outcome,
                ]
            )

    def reset(self):
        self.time_old = 0
        self.episodes_passed = 0
        self.iterations_passed = 0


logger3WRobot = Logger(
    ["x [m]", "y [m]", "angle [rad]", "v [m/s]", "omega [rad/s]"], ["F [N]", "M [N m]"]
)

logger3WRobotNI = Logger(
    ["x [m]", "y [m]", "angle [rad]"],
    ["v [m/s]", "omega [rad/s]"],
    ["8.3f", "8.3f", "8.3f", "8.3f", "8.1f", "8.1f", "8.3f", "8.3f"],
)

logger2Tank = Logger(
    ["h1", "h2"], ["p"], ["8.1f", "8.4f", "8.4f", "8.4f", "8.4f", "8.2f"]
)

loggerInvertedPendulum = Logger(["angle [rad]", "angle_dot [rad/s]"], ["M [N m]"])
