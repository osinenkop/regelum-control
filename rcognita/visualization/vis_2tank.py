import numpy as np
from .animator import (
    update_line,
    update_text,
    reset_line,
    init_data_cursor,
    Dashboard,
    Animator,
)
from mpldatacursor import datacursor
from collections import namedtuple
import matplotlib.patheffects as PathEffects
from ..__utilities import rc
import matplotlib.pyplot as plt


class Animator2Tank(Animator):
    """
    Animator class for a 2-tank system.

    """

    def __init__(self, objects=None, pars=None):
        self.objects = objects
        self.pars = pars

        # Unpack entities
        (
            self.simulator,
            self.system,
            self.nominal_controller,
            self.controller,
            self.datafiles,
            self.logger,
        ) = self.objects

        (
            state_init,
            action_init,
            time_start,
            time_final,
            state_full_init,
            control_mode,
            action_manual,
            action_min,
            action_max,
            no_print,
            is_log,
            is_playback,
            running_obj_init,
            level_target,
        ) = self.pars

        # Store some parameters for later use
        self.time_start = time_start
        self.state_full_init = state_full_init
        self.time_final = time_final
        self.control_mode = control_mode
        self.action_manual = action_manual
        self.no_print = no_print
        self.is_log = is_log
        self.is_playback = is_playback

        self.level_target = level_target

        h1_0 = state_init[0]
        h2_0 = state_init[1]
        p0 = action_init

        plt.close("all")

        self.fig_sim = plt.figure(figsize=(10, 10))

        # h1, h2 plot
        self.axs_sol = self.fig_sim.add_subplot(
            221,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(-2, 2),
            xlabel="Time [s]",
            title="Pause - space, q - quit, click - data cursor",
        )
        self.axs_sol.plot([time_start, time_final], [0, 0], "k--", lw=0.75)  # Help line
        self.axs_sol.plot(
            [time_start, time_final], [level_target[0], level_target[0]], "b--", lw=0.75
        )  # Help line (target)
        self.axs_sol.plot(
            [time_start, time_final], [level_target[1], level_target[1]], "r--", lw=0.75
        )  # Help line (target)
        (self.line_h1,) = self.axs_sol.plot(
            time_start, h1_0, "b-", lw=0.5, label=r"$h_1$"
        )
        (self.line_h2,) = self.axs_sol.plot(
            time_start, h2_0, "r-", lw=0.5, label=r"$h_2$"
        )
        self.axs_sol.legend(fancybox=True, loc="upper right")
        self.axs_sol.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )

        # Cost
        if is_playback:
            running_objective = running_obj_init
        else:
            observation_init = self.system.out(state_init)
            running_objective = self.controller.running_objective(
                observation_init, action_init
            )

        self.axs_cost = self.fig_sim.add_subplot(
            223,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(0, 1e4 * running_objective),
            yscale="symlog",
            xlabel="Time [s]",
        )

        text_outcome = r"$\int \mathrm{{stage\,obj.}} \,\mathrm{{d}}t$ = {outcome:2.3f}".format(
            outcome=0
        )
        self.text_outcome_handle = self.fig_sim.text(
            0.05,
            0.5,
            text_outcome,
            horizontalalignment="left",
            verticalalignment="center",
        )
        (self.line_running_obj,) = self.axs_cost.plot(
            time_start, running_objective, "r-", lw=0.5, label="Running obj."
        )
        (self.line_outcome,) = self.axs_cost.plot(
            time_start,
            0,
            "g-",
            lw=0.5,
            label=r"$\int \mathrm{stage\,obj.} \,\mathrm{d}t$",
        )
        self.axs_cost.legend(fancybox=True, loc="upper right")

        # Control
        self.axs_action = self.fig_sim.add_subplot(
            222,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(action_min - 0.1, action_max + 0.1),
            xlabel="Time [s]",
        )
        self.axs_action.plot(
            [time_start, time_final], [0, 0], "k--", lw=0.75
        )  # Help line
        (self.line_action,) = self.axs_action.plot(time_start, p0, lw=0.5, label="p")
        self.axs_action.legend(fancybox=True, loc="upper right")

        # Pack all lines together
        cLines = namedtuple(
            "lines",
            ["line_h1", "line_h2", "line_running_obj", "line_outcome", "line_action"],
        )
        self.lines = cLines(
            line_h1=self.line_h1,
            line_h2=self.line_h2,
            line_running_obj=self.line_running_obj,
            line_outcome=self.line_outcome,
            line_action=self.line_action,
        )

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

    def set_sim_data(self, ts, h1s, h2s, ps, rs, outcomes):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """
        self.ts, self.h1s, self.h2s, self.ps = ts, h1s, h2s, ps
        self.rs, self.outcomes = rs, outcomes
        self.curr_step = 0

    def update_sim_data_row(self):
        self.time = self.ts[self.curr_step]
        self.state_full = np.array([self.h1s[self.curr_step], self.h2s[self.curr_step]])
        self.running_objective = self.rs[self.curr_step]
        self.outcome = self.outcomes[self.curr_step]
        self.action = np.array([self.ps[self.curr_step]])

        self.curr_step = self.curr_step + 1

    def init_anim(self):
        state_init, *_ = self.pars

        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]

    def animate(self, k):

        if self.is_playback:
            self.update_sim_data_row()
            time = self.time
            state_full = self.state_full
            action = self.action
            running_objective = self.running_objective
            outcome = self.outcome

        else:
            self.simulator.do_sim_step()

            time, state, observation, state_full = self.simulator.get_sim_step_data()

            action = self.controller.compute_action(time, observation)

            self.system.receive_action(action)
            self.controller.update_outcome(observation, action)

            running_objective = self.controller.running_objective(observation, action)
            outcome = self.controller.outcome

        h1 = state_full[0]
        h2 = state_full[1]
        p = action

        if not self.no_print:
            self.logger.print_sim_step(time, h1, h2, p, running_objective, outcome)

        if self.is_log:
            self.logger.log_data_row(
                self.datafile_curr, time, h1, h2, p, running_objective, outcome
            )

        # # Solution
        update_line(self.line_h1, time, h1)
        update_line(self.line_h2, time, h2)

        # Cost
        update_line(self.line_running_obj, time, running_objective)
        update_line(self.line_outcome, time, outcome)
        text_outcome = r"$\int \mathrm{{stage\,obj.}} \,\mathrm{{d}}t$ = {outcome:2.1f}".format(
            outcome=np.squeeze(np.array(outcome))
        )
        update_text(self.text_outcome_handle, text_outcome)

        # Control
        update_line(self.line_action, time, p)

        # Run done
        if time >= self.time_final:
            if not self.no_print:
                print(
                    ".....................................Run {run:2d} done.....................................".format(
                        run=self.run_curr
                    )
                )

            self.run_curr += 1


            if self.is_log:
                self.datafile_curr = self.datafiles[self.run_curr - 1]

            # Reset simulator
            self.simulator.reset()

            # Reset controller
            if self.control_mode > 0:
                self.controller.reset(self.time_start)
            else:
                self.nominal_controller.reset(self.time_start)

            outcome = 0

            reset_line(self.line_h1)
            reset_line(self.line_h1)
            reset_line(self.line_action)
            reset_line(self.line_running_obj)
            reset_line(self.line_outcome)

            # for item in self.lines:
            #     if item != self.line_trajectory:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)

            # update_line(self.line_h1, np.nan)
