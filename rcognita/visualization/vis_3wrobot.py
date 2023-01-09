import numpy as np
import numpy.linalg as la
from .animator import (
    update_line,
    update_text,
    reset_line,
    Animator,
    RobotMarker,
    Dashboard,
    init_data_cursor,
)
from mpldatacursor import datacursor
from collections import namedtuple
from ..__utilities import rc
import matplotlib.pyplot as plt


class RobotTrackingDasboard(Dashboard):
    def __init__(
        self, time_start, xMax, xMin, yMax, yMin, xCoord0, yCoord0, angle_deg0, scenario
    ):
        super().__init__()
        self.time_start = time_start
        self.xMax = xMax
        self.xMin = xMin
        self.yMax = yMax
        self.yMin = yMin
        self.angle_deg0 = angle_deg0
        self.xCoord0 = xCoord0
        self.yCoord0 = yCoord0
        self.scenario = scenario

    def init_dashboard(self):
        self.axes_3wrobot = plt.gca()

        self.axes_3wrobot.set_xlim(self.xMin, self.xMax)
        self.axes_3wrobot.set_ylim(self.yMin, self.yMax)
        self.axes_3wrobot.set_xlabel("x [m]")
        self.axes_3wrobot.set_ylabel("y [m]")
        self.axes_3wrobot.set_title("Pause - space, q - quit, click - data cursor")

        self.axes_3wrobot.set_aspect("equal", adjustable="box")
        self.axes_3wrobot.plot(
            [self.xMin, self.xMax], [0, 0], "k--", lw=0.75
        )  # Help line
        self.axes_3wrobot.plot(
            [0, 0], [self.yMin, self.yMax], "k--", lw=0.75
        )  # Help line
        (self.line_trajectory,) = self.axes_3wrobot.plot(
            self.xCoord0, self.yCoord0, "b--", lw=0.5
        )
        self.artists.append(self.line_trajectory)
        self.robot_marker = RobotMarker(angle=self.angle_deg0)
        text_time = f"Time = {self.time_start:2.3f}"
        self.text_time_handle = self.axes_3wrobot.text(
            0.05,
            0.95,
            text_time,
            horizontalalignment="left",
            verticalalignment="center",
            transform=self.axes_3wrobot.transAxes,
        )
        self.artists.append(self.text_time_handle)

        self.axes_3wrobot.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )
        self.scatter_sol = self.axes_3wrobot.scatter(
            self.xCoord0, self.yCoord0, marker=self.robot_marker.marker, s=400, c="b"
        )

    def perform_step_update(self):
        state = self.scenario.state_full
        time = self.scenario.time

        xCoord = state[0]
        yCoord = state[1]
        angle = state[2]
        angle_deg = angle / np.pi * 180
        text_time = f"Time = {time:2.3f}"

        update_text(self.text_time_handle, text_time)
        update_line(self.line_trajectory, xCoord, yCoord)
        self.robot_marker.rotate(angle_deg)  # Rotate the robot on the plot
        self.scatter_sol.remove()
        self.scatter_sol = self.axes_3wrobot.scatter(
            xCoord, yCoord, marker=self.robot_marker.marker, s=400, c="b"
        )


class SolutionDashboard(Dashboard):
    def __init__(
        self,
        time_start,
        time_final,
        xMax,
        xMin,
        yMax,
        yMin,
        xCoord0,
        yCoord0,
        angle0,
        scenario,
    ):
        super().__init__()
        self.time_start = time_start
        self.time_final = time_final
        self.xMax = xMax
        self.xMin = xMin
        self.yMax = yMax
        self.yMin = yMin
        self.xCoord0 = xCoord0
        self.yCoord0 = yCoord0
        self.angle0 = angle0
        self.scenario = scenario

    def init_dashboard(self):
        self.axes_solution = plt.gca()

        self.axes_solution.autoscale(False)
        self.axes_solution.set_xlim(self.time_start, self.time_final)
        self.axes_solution.set_ylim(
            2 * np.min([self.xMin, self.yMin]), 2 * np.max([self.xMax, self.yMax]),
        )
        self.axes_solution.set_xlabel("Time [s]")

        self.axes_solution.plot(
            [self.time_start, self.time_final], [0, 0], "k--", lw=0.75
        )  # Help line
        (self.line_norm,) = self.axes_solution.plot(
            self.time_start,
            la.norm([self.xCoord0, self.yCoord0]),
            "b-",
            lw=0.5,
            label=r"$\Vert(x,y)\Vert$ [m]",
        )
        self.artists.append(self.line_norm)
        (self.line_angle,) = self.axes_solution.plot(
            self.time_start, self.angle0, "r-", lw=0.5, label=r"$\angle$ [rad]"
        )
        self.artists.append(self.line_angle)
        self.axes_solution.legend(fancybox=True, loc="upper right")
        self.axes_solution.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )

    def perform_step_update(self):
        state = self.scenario.state_full
        time = self.scenario.time

        xCoord = state[0]
        yCoord = state[1]
        angle = state[2]
        # # Solution
        update_line(self.line_norm, time, la.norm([xCoord, yCoord]))
        update_line(self.line_angle, time, np.squeeze(angle))


class CostDashboard(Dashboard):
    def __init__(self, time_start, time_final, running_objective_init, scenario):
        super().__init__()
        self.time_start = time_start
        self.time_final = time_final
        self.running_objective_init = running_objective_init
        self.scenario = scenario

    def init_dashboard(self):
        self.axes_cost = plt.gca()

        self.axes_cost.set_xlim(self.time_start, self.time_final)
        self.axes_cost.set_ylim(0, 1e4)
        self.axes_cost.set_yscale("symlog")
        self.axes_cost.set_xlabel("Time [s]")
        self.axes_cost.autoscale(False)

        text_outcome = r"$\int \mathrm{{running\,obj.}} \,\mathrm{{d}}t$ = {outcome:2.3f}".format(
            outcome=0
        )
        self.text_outcome_handle = self.axes_cost.text(
            0.05,
            0.5,
            text_outcome,
            horizontalalignment="left",
            verticalalignment="center",
        )
        (self.line_running_obj,) = self.axes_cost.plot(
            self.time_start,
            self.running_objective_init,
            "r-",
            lw=0.5,
            label="Running obj.",
        )
        (self.line_outcome,) = self.axes_cost.plot(
            self.time_start,
            0,
            "g-",
            lw=0.5,
            label=r"$\int \mathrm{running\,obj.} \,\mathrm{d}t$",
        )
        self.artists.append(self.line_running_obj)
        self.artists.append(self.line_outcome)
        self.axes_cost.legend(fancybox=True, loc="upper right")

    def perform_step_update(self):
        time = self.scenario.time
        running_objective_value = np.squeeze(self.scenario.running_objective_value)
        outcome = self.scenario.outcome

        update_line(self.line_running_obj, time, running_objective_value)
        update_line(self.line_outcome, time, outcome)
        text_outcome = r"$\int \mathrm{{running\,obj.}} \,\mathrm{{d}}t$ = {outcome:2.1f}".format(
            outcome=np.squeeze(np.array(outcome))
        )
        update_text(self.text_outcome_handle, text_outcome)


class ControlDashboard(Dashboard):
    def __init__(self, time_start, time_final, F_min, F_max, M_min, M_max, scenario):
        super().__init__()
        self.time_start = time_start
        self.time_final = time_final
        self.F_min = F_min
        self.F_max = F_max
        self.M_min = M_min
        self.M_max = M_max
        self.scenario = scenario

    def init_dashboard(self):
        self.axis_action = plt.gca()

        self.axis_action.set_xlim(self.time_start, self.time_final)
        self.axis_action.set_ylim(
            1.1 * np.min([self.F_min, self.M_min]),
            1.1 * np.max([self.F_max, self.M_max]),
        )
        self.axis_action.set_xlabel("Time [s]")
        self.axis_action.set_ylabel("Control")
        self.axis_action.autoscale(False)

        self.axis_action.plot(
            [self.time_start, self.time_final], [0, 0], "k--", lw=0.75
        )  # Help line
        self.lines_action = self.axis_action.plot(
            self.time_start, rc.force_column(self.scenario.action_init).T, lw=0.5
        )
        self.axis_action.legend(
            iter(self.lines_action),
            ("F [N]", "M [Nm]"),
            fancybox=True,
            loc="upper right",
        )
        self.artists.extend(self.lines_action)

    def perform_step_update(self):
        # Control
        action = np.squeeze(self.scenario.action)
        time = self.scenario.time

        for (line, action_single) in zip(self.lines_action, action):
            update_line(line, time, action_single)


class ControlDashboardNI(Dashboard):
    def __init__(
        self, time_start, time_final, v_min, v_max, omega_min, omega_max, scenario
    ):
        super().__init__()

        self.time_start = time_start
        self.time_final = time_final
        self.v_min = v_min
        self.v_max = v_max
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.scenario = scenario

    def init_dashboard(self):
        self.axis_action = plt.gca()

        self.axis_action.set_xlim(self.time_start, self.time_final)
        self.axis_action.set_ylim(
            1.1 * np.min([self.v_min, self.omega_min]),
            1.1 * np.max([self.v_max, self.omega_max]),
        )
        self.axis_action.set_xlabel("Time [s]")
        self.axis_action.set_ylabel("Control")
        self.axis_action.autoscale(False)

        self.axis_action.plot(
            [self.time_start, self.time_final], [0, 0], "k--", lw=0.75
        )  # Help line
        self.lines_action = self.axis_action.plot(
            self.time_start, rc.force_column(self.scenario.action_init).T, lw=0.5
        )
        self.axis_action.legend(
            iter(self.lines_action),
            ("v [m/s]", r"$\omega$ [rad/s]"),
            fancybox=True,
            loc="upper right",
        )
        self.artists.extend(self.lines_action)

    def perform_step_update(self):
        # Control
        action = self.scenario.action
        time = self.scenario.time

        for (line, action_single) in zip(self.lines_action, np.array(action)):
            update_line(line, time, action_single)


class Animator3WRobot(Animator):
    """
    Animator class for a 3-wheel robot with dynamic actuators.

    """

    def __init__(self, scenario=None, subplot_grid_size=None):
        if subplot_grid_size is None:
            subplot_grid_size = [2, 2]
        super().__init__(subplot_grid_size=subplot_grid_size)
        self.scenario = scenario
        self.__dict__.update(scenario.__dict__)

        (
            state_full_init,
            xMin,
            xMax,
            yMin,
            yMax,
            F_min,
            M_min,
            F_max,
            M_max,
            running_obj_init,
        ) = (
            self.scenario.simulator.state_full_init,
            -10,
            10,
            -10,
            10,
            self.scenario.controller.action_bounds[0][0],
            self.scenario.controller.action_bounds[0][1],
            self.scenario.controller.action_bounds[1][0],
            self.scenario.controller.action_bounds[1][1],
            0,
        )
        self.sampling_time = self.scenario.controller.sampling_time

        # Store some parameters for later use
        self.time_old = 0
        self.outcome = 0
        self.state_full_init = state_full_init

        xCoord0 = self.state_init[0]
        yCoord0 = self.state_init[1]
        angle0 = self.state_init[2]
        angle_deg0 = angle0 / 2 / np.pi

        ########### SUBPLOT 1  --------- PENDULUM TRACKING ###########################
        robot_tracking_dasboard = RobotTrackingDasboard(
            self.time_start,
            xMax,
            xMin,
            yMax,
            yMin,
            xCoord0,
            yCoord0,
            angle_deg0,
            self.scenario,
        )
        ########### SUBPLOT 2  --------- STEP-BY-STEP-SOLUTION #######################
        solution_dashboard = SolutionDashboard(
            self.time_start,
            self.time_final,
            xMax,
            xMin,
            yMax,
            yMin,
            xCoord0,
            yCoord0,
            angle0,
            self.scenario,
        )

        ########### SUBPLOT 3  --------- COST ########################################

        if self.is_playback:
            running_objective = running_obj_init
        else:
            observation_init = self.system.out(self.state_init)
            running_objective = self.running_objective(
                observation_init, self.action_init
            )

        cost_dashboard = CostDashboard(
            self.time_start, self.time_final, running_objective, self.scenario
        )
        ########### SUBPLOT 4  --------- CONTROL #####################################

        control_dashboard = ControlDashboard(
            self.time_start, self.time_final, F_min, F_max, M_min, M_max, self.scenario
        )
        ##############################################################################

        self.collect_dashboards(
            robot_tracking_dasboard,
            solution_dashboard,
            cost_dashboard,
            control_dashboard,
        )


class Animator3WRobotNI(Animator):
    """
    Animator class for a 3-wheel robot with static actuators.

    """

    def __init__(self, scenario=None, subplot_grid_size=None):
        if subplot_grid_size is None:
            subplot_grid_size = [2, 2]
        super().__init__(subplot_grid_size=subplot_grid_size)
        self.scenario = scenario
        self.__dict__.update(scenario.__dict__)
        self.sampling_time = self.scenario.controller.sampling_time

        # Unpack entities
        (
            state_full_init,
            xMin,
            xMax,
            yMin,
            yMax,
            v_min,
            omega_min,
            v_max,
            omega_max,
            running_obj_init,
        ) = (
            self.scenario.simulator.state_full_init,
            -10,
            10,
            -10,
            10,
            self.scenario.controller.action_bounds[0][0],
            self.scenario.controller.action_bounds[0][1],
            self.scenario.controller.action_bounds[1][0],
            self.scenario.controller.action_bounds[1][1],
            0,
        )
        # Store some parameters for later use
        self.time_old = 0
        self.outcome = 0
        self.state_full_init = state_full_init

        xCoord0 = self.state_init[0]
        yCoord0 = self.state_init[1]
        angle0 = self.state_init[2]
        angle_deg0 = angle0 / 2 / np.pi

        ########### SUBPLOT 1  --------- PENDULUM TRACKING ###########################
        robot_tracking_dasboard = RobotTrackingDasboard(
            self.time_start,
            xMax,
            xMin,
            yMax,
            yMin,
            xCoord0,
            yCoord0,
            angle_deg0,
            self.scenario,
        )
        ########### SUBPLOT 2  --------- STEP-BY-STEP-SOLUTION #######################
        solution_dashboard = SolutionDashboard(
            self.time_start,
            self.time_final,
            xMax,
            xMin,
            yMax,
            yMin,
            xCoord0,
            yCoord0,
            angle0,
            self.scenario,
        )

        ########### SUBPLOT 3  --------- COST ########################################

        if self.is_playback:
            running_objective = running_obj_init
        else:
            observation_init = self.system.out(self.state_init)
            running_objective = self.running_objective(
                observation_init, self.action_init
            )

        cost_dashboard = CostDashboard(
            self.time_start, self.time_final, running_objective, self.scenario
        )

        ########### SUBPLOT 4  --------- CONTROL #####################################

        control_dashboard = ControlDashboardNI(
            self.time_start,
            self.time_final,
            v_min,
            v_max,
            omega_min,
            omega_max,
            self.scenario,
        )

        ##############################################################################

        self.collect_dashboards(
            robot_tracking_dasboard,
            solution_dashboard,
            cost_dashboard,
            control_dashboard,
        )
