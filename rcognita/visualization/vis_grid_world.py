import numpy as np
from .animator import update_line, Animator
import matplotlib.patheffects as PathEffects
from ..__utilities import rc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class AnimatorGridWorld(Animator):
    def __init__(
        self,
        reward_cell_xy,
        starting_cell_xy,
        punishment_cells,
        scenario,
        N_iterations=50,
    ):
        length = 3
        self.actions_map = {
            0: np.array([0.01 * length, 0]),
            1: np.array([-0.01 * length, 0]),
            2: np.array([0, 0.01 * length]),
            3: np.array([0, -0.01 * length]),
            4: np.array([0, 0]),
        }
        self.actor = scenario.actor
        self.critic = scenario.critic
        self.starting_cell_xy = starting_cell_xy
        self.reward_cell_xy = reward_cell_xy
        self.punishment_cells = punishment_cells
        self.scenario = scenario
        self.N_iterations = N_iterations

        self.colormap = plt.get_cmap("RdYlGn_r")

        self.fig_sim = plt.figure(figsize=(10, 10))

        self.ax = self.fig_sim.add_subplot(
            211,
            xlabel="red-green gradient corresponds to value (except for target and black cells).\nStarting cell is yellow",
            ylabel="",
            xlim=(0, 1.015),
            ylim=(-0.015, 1),
            facecolor="grey",
        )
        self.ax.set_title(label="Pause - space, q - quit", pad="20.0")
        self.ax.set_aspect("equal")
        self.arrows_patch_pack, self.rect_patch_pack, self.text_pack = self.create_grid(
            self.ax
        )
        normalize = mcolors.Normalize(vmin=70, vmax=100)
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=self.colormap)
        scalarmappaple.set_array(self.critic.model.weights)
        self.colorbar = plt.colorbar(scalarmappaple)

        self.ax_value_plot = self.fig_sim.add_subplot(
            212,
            autoscale_on=False,
            xlabel="Iteration",
            ylabel="Value",
            xlim=(0, self.N_iterations),
            ylim=(-100, 100),
            title="Plot of the value at starting cell",
        )
        (self.line_value,) = self.ax_value_plot.plot(
            0,
            self.critic.model.weights[
                self.starting_cell_xy[0], self.starting_cell_xy[1]
            ],
            "g-",
            lw=1.5,
            label="Value",
        )
        plt.axhline(76, c="r", linestyle="--", label="Optimal value")
        plt.legend()

    def update_grid(self, iter):
        table = self.critic.model.weights
        shape = table.shape
        lenght = shape[0]
        width = shape[1]
        for i in range(lenght):
            for j in range(width):
                val = table[i, j]
                action = self.actor.model.weights[i, j]
                table_range = np.ptp(np.fmax(table, 70))
                color = self.colormap((val - np.max([np.min(table), 70])) / table_range)
                rectangle = self.rect_patch_pack[i * width + j]
                arr_x, arr_y = self.map_action2arrow(action, rectangle)

                arrow = self.arrows_patch_pack[i * width + j]
                arrow.set_positions(
                    (arr_x, arr_y),
                    (
                        arr_x + self.actions_map[action][0],
                        arr_y + self.actions_map[action][1],
                    ),
                )
                text = self.text_pack[i * width + j]
                if (
                    self.reward_cell_xy != [i, j]
                    and self.starting_cell_xy != [i, j]
                    and [i, j] not in self.punishment_cells
                ):
                    rectangle.set_facecolor(color)
                text.set_text(str(int(val)))

                if self.starting_cell_xy == [i, j]:
                    update_line(self.line_value, iter, val)

    def create_grid(self, ax, space=0.01):
        table = self.critic.model.weights
        shape = table.shape
        lenght = shape[0]
        width = shape[1]
        rect_patch_pack = []
        arrows_patch_pack = []
        text_pack = []

        for i in range(lenght):
            for j in range(width):
                val = table[i, j]
                table_range = 200
                color = self.colormap(
                    ((val - np.min(table)) / table_range)
                    if np.abs(table_range) > 1e-3
                    else 1
                )
                if self.reward_cell_xy == [i, j]:
                    rectangle = patches.Rectangle(
                        (j / width + space, 1 - (i + 1) / lenght),
                        1 / lenght - space,
                        1 / width - space,
                        linewidth=2,
                        edgecolor="g",
                        facecolor="r",
                    )
                elif self.starting_cell_xy == [i, j]:
                    rectangle = patches.Rectangle(
                        (j / width + space, 1 - (i + 1) / lenght),
                        1 / lenght - space,
                        1 / width - space,
                        linewidth=2,
                        edgecolor="b",
                        facecolor="yellow",
                    )
                elif [i, j] in self.punishment_cells:
                    rectangle = patches.Rectangle(
                        (j / width + space, 1 - (i + 1) / lenght),
                        1 / lenght - space,
                        1 / width - space,
                        linewidth=2,
                        edgecolor="black",
                        facecolor="grey",
                    )
                else:
                    rectangle = patches.Rectangle(
                        (j / width + space, 1 - (i + 1) / lenght),
                        1 / lenght - space,
                        1 / width - space,
                        linewidth=2,
                        edgecolor="g",
                        facecolor=color,
                    )
                rx, ry = rectangle.get_xy()
                cx = rx + rectangle.get_width() / 2.0
                cy = ry + rectangle.get_height() / 2.0
                text = ax.text(
                    cx,
                    cy,
                    str(np.floor(val)),
                    color="black",
                    weight="bold",
                    fontsize=7,
                    ha="center",
                    va="center",
                )
                text.set_path_effects(
                    [PathEffects.withStroke(linewidth=2, foreground="w")]
                )
                ax.add_patch(rectangle)
                rect_patch_pack.append(rectangle)
                text_pack.append(text)
                ax.set(xticks=[], yticks=[])
                action = self.actor.model.weights[i, j]
                arr_x, arr_y = self.map_action2arrow(action, rectangle)

                pos_head = self.actions_map[action]
                arrowstyle = patches.ArrowStyle.Fancy(
                    head_length=0.4, head_width=1, tail_width=4
                )
                if self.reward_cell_xy == [i, j]:
                    arrowstyle = patches.ArrowStyle.Fancy(
                        head_length=0, head_width=1, tail_width=1
                    )
                arrow = patches.FancyArrowPatch(
                    (arr_x, arr_y),
                    (arr_x + pos_head[0], arr_y + pos_head[1]),
                    arrowstyle=arrowstyle,
                )
                # arrow.set_arrowstyle("fancy", head_length=0.05)
                ax.add_patch(arrow)
                arrows_patch_pack.append(arrow)
                if i == 0:
                    text = ax.text(cx, 1.03, f"{j}", ha="center", va="center")
                if j == 0:
                    text = ax.text(-0.03, cy, f"{i}", ha="center", va="center")

        return arrows_patch_pack, rect_patch_pack, text_pack

    def map_action2arrow(self, action, rectangle):
        rx, ry = rectangle.get_xy()
        if action == 0:
            arr_x = rx + rectangle.get_width() * (7 / 10)
            arr_y = ry + rectangle.get_height() / 2
        elif action == 1:
            arr_x = rx + rectangle.get_width() * (3 / 10)
            arr_y = ry + rectangle.get_height() / 2
        elif action == 2:
            arr_x = rx + rectangle.get_width() / 2
            arr_y = ry + rectangle.get_height() * (7 / 10)
        elif action == 3:
            arr_x = rx + rectangle.get_width() / 2
            arr_y = ry + rectangle.get_height() * (3 / 10)
        elif action == 4:
            arr_x = rx + rectangle.get_width() / 2
            arr_y = ry + rectangle.get_height() / 2

        return arr_x, arr_y

    def animate(self, k):
        self.scenario.step()
        self.update_grid(k)
