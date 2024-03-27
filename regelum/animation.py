"""Callbacks that create, display and store animation according to dynamic data."""

import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from copy import copy
from multiprocessing import Process
from unittest.mock import Mock
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

matplotlib.use("Agg")
import matplotlib.animation
import matplotx.styles
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.transforms import Affine2D

import regelum
import os


import matplotlib.pyplot as plt


from pathlib import Path

import regelum.__internal.base

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends import backend_qt5agg  # e.g.

from svgpathtools import svg2paths
from svgpath2mpl import parse_path

import regelum.callback as callback


plt.rcParams["animation.frame_format"] = "svg"  # VERY important

import matplotlib.style as mplstyle

plt.style.use(matplotx.styles.dracula)
# plt.style.use('fast')
# plt.rcParams['path.simplify'] = True

# plt.rcParams['path.simplify_threshold'] = 1.0


class AnimationCallback(callback.Callback, ABC):
    """Callback (base) responsible for rendering animated visualizations of the experiment."""

    _frames = 100
    _is_global = False

    @classmethod
    def _compose(cls_left, cls_right):
        animations = []
        for cls in [cls_left, cls_right]:
            if cls is None:
                continue
            if issubclass(cls, ComposedAnimationCallback) and not issubclass(
                cls, DeferredComposedAnimation
            ):
                assert (
                    cls is not ComposedAnimationCallback
                ), "Adding a composed animation in this way is ambiguous."
                animations += cls._animations
            elif issubclass(cls, AnimationCallback):
                animations.append(cls)
            else:
                raise TypeError(
                    "An animation class can only be added with another animation class or None."
                )

        class Composed(ComposedAnimationCallback):
            _animations = animations

            def __init__(self, **kwargs):
                super().__init__(*self._animations, **kwargs)

        return Composed

    def __init__(self, *args, interactive=None, fig=None, ax=None, mng=None, **kwargs):
        """Initialize an instance of AnimationCallback."""
        super().__init__(*args, **kwargs)
        # assert (
        #    self.attachee is not None
        # ), "An animation callback can only be instantiated via attachment, however an attempt to instantiate it otherwise was made."
        self.frame_data = []
        # matplotlib.use('Qt5Agg', force=True)

        self.interactive_mode = (
            self._metadata["argv"].interactive if interactive is None else interactive
        )
        if self.interactive_mode:
            matplotlib.use("Agg")
            self.fig = Figure(figsize=(10, 10))
            canvas = FigureCanvas(self.fig)

            self.ax = canvas.figure.add_subplot(111)
            self.mng = backend_qt5agg.new_figure_manager_given_figure(1, self.fig)
            self.setup()
        else:
            self.mng = mng

            self.fig, self.ax = fig, ax
            if isinstance(self.ax, SubplotSpec):
                self.ax = fig.add_subplot(self.ax)
        self.save_directory = Path(
            f".callbacks/{self.__class__.__name__}"
            + (f"@{self.attachee.__name__}" if self.attachee is not None else "")
        ).resolve()
        self.saved_counter = 0
        self.skip_frames = self._metadata["skip_frames"]
        self.counter = 0

    def __del__(self):
        if hasattr(self, "updater"):
            self.canvas_shm["canvas"] = None
            self.updater.join()

    @abstractmethod
    def setup(self):
        pass #self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    def get_save_directory(self):
        return str(self.save_directory)

    def add_frame(self, **frame_datum):
        if self.counter is not None:
            self.counter += 1
            if (self.counter - 1) % self.skip_frames:
                return
        self.frame_data.append(frame_datum)
        if hasattr(self, "interactive_status"):
            self.interactive_status["frame"] = frame_datum

        if self.interactive_mode:
            self.lim(frame_idx=len(self.frame_data) - 1)
            self.construct_frame(**frame_datum)
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            # while self.fig.paused:
            #    self.fig.canvas.flush_events()

    def __getstate__(self):
        state = copy(self.__dict__)
        del state["mng"]

        del state["attachee"]

        if "canvas" in state:
            del state["canvas"]

        # del state["log"]
        # del state["exception"]
        # del state["mng"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mng = Mock()
        self.attachee = Mock()

    def reset(self):
        self.frame_data = []
        if self.interactive_mode:
            # self.mng.window.close()
            pass
        self.ax.clear()
        self.setup()
        # self.__init__(attachee=self.attachee, interactive=self.interactive_mode)

    @abstractmethod
    def construct_frame(self, **frame_datum):
        pass

    def animate(self, frames=None):
        temp_fig, temp_ax = self.fig, self.ax
        plt.ioff()
        self.fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(self.fig)
        self.ax = canvas.figure.add_subplot(111)
        if frames is None:
            frames = self.__class__._frames
        if frames == "all":
            frames = len(self.frame_data)
        elif isinstance(frames, int):
            frames = max(min(frames, len(self.frame_data)), 0)
        elif isinstance(frames, float) and 0 <= frames <= 1:
            frames = int(frames * len(self.frame_data))
        else:
            raise ValueError(
                'animate accepts an int, a float or "all", but instead a different value was provided.'
            )
        if not hasattr(self, "trajectory"):

            def animation_update(i):
                j = int((i / frames) * len(self.frame_data) + 0.5)
                self.lim(frame_idx=j)
                return self.construct_frame(**self.frame_data[j])

        else:
            trajectory_xs = self.trajectory_xs
            trajectory_ys = self.trajectory_ys

            def animation_update(i):
                j = int((i / frames) * len(self.frame_data) + 0.5)
                self.lim(frame_idx=j)
                return self.construct_frame(
                    trajectory_xs=trajectory_xs[:j],
                    trajectory_ys=trajectory_ys[:j],
                    **self.frame_data[j],
                )

        self.setup()

        anim = matplotlib.animation.FuncAnimation(
            self.fig,
            animation_update,
            frames=frames,
            interval=30,
            blit=True,
            repeat=False,
            #init_func=self.setup
        )
        res = anim.to_jshtml()
        plt.close(self.fig)
        self.fig, self.ax = temp_fig, temp_ax
        return res

    def on_launch(self):
        os.mkdir(self.get_save_directory())

    def animate_and_save(self, frames=None, name=None):
        if name is None:
            name = str(self.saved_counter)
            self.saved_counter += 1
        animation = self.animate(frames=frames)
        dir = self.get_save_directory()
        with open(dir + "/" + name + ".html", "w") as f:
            f.write(
                f"<html><head><title>{self.__class__.__name__}: {name}</title></head><body>{animation}</body></html>"
            )

    @classmethod
    def lim_from_reference(cls, x, y, extra_margin=0, equal=True):
        x = np.array(x)
        y = np.array(y)
        if x.size == 0:
            x = np.zeros(1)
        if y.size == 0:
            y = np.zeros(1)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1
        y_min, y_max = y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1
        left = x_min - extra_margin
        width = max(x_max - x_min, y_max - y_min) if equal else x_max - x_min
        right = x_min + width + extra_margin
        bottom = y_min - extra_margin
        height = max(x_max - x_min, y_max - y_min) if equal else y_max - y_min
        top = y_min + height + extra_margin
        return left, right, bottom, top

    def lim(self, width=None, height=None, center=None, extra_margin=0.0, **kwargs):
        if width is not None or height is not None:
            if center is None:
                center = [0.0, 0.0]
            if width is None:
                width = height
            if height is None:
                height = width
            self.ax.set_xlim(
                center[0] - width / 2 - extra_margin,
                center[0] + width / 2 + extra_margin,
            )
            self.ax.set_ylim(
                center[1] - height / 2 - extra_margin,
                center[1] - height / 2 + extra_margin,
            )
        else:
            raise ValueError("No axis limits known for animation.")

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        self.counter = 0
        # self.animate_and_save(name=str(episode_number))
        self.reset()

    def install_temp_axis(self, fig, ax, interactive=None):
        self._old_ax = self.ax
        self._old_fig = self.fig
        self._old_interactive_mode = self.interactive_mode
        self.fig = fig
        self.ax = ax
        if isinstance(self.ax, SubplotSpec):
            self.ax = fig.add_subplot(self.ax)
        if interactive is not None:
            self.interactive_mode = interactive

    def pop_axis(self):
        self.fig = self._old_fig
        self.ax = self._old_ax
        self.interactive_mode = self._old_interactive_mode
        del self._old_fig
        del self._old_ax
        del self._old_interactive_mode

    @property
    def latest_frame(self):
        return self.frame_data[-1]

    @property
    def latest_frame_idx(self):
        return len(self.frame_data) - 1


class PausableFigure(Figure):
    def __init__(self, *args, log=print, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = False

        def on_press(event):
            if event.key == " ":
                self.paused = not self.paused
            log("Paused." if self.paused else "Resumed.")

        self.canvas.mpl_connect("key_press_event", on_press)


def title_except_units(s):
    words = s[:s.find('[')]
    units = s[s.find('['):]
    return (words.title() + units).replace("Velocity", "Vel.")

class ComposedAnimationCallback(AnimationCallback):
    """An animation callback capable of incoroporating several other animation callbacks in such a way that the respective plots are distributed between axes of a single figure."""

    def __init__(
        self, *animations, fig=None, ax=None, mng=None, mode="square", **kwargs
    ):
        """Initialize an instance of ComposedAnimationCallback.

        Args:
            *animations (*List[AnimationCallback]): animation classes to be composed
            **kwargs (**Dict[str, Any]): keyword arguments to be passed to __init__ of said
                animations and the base class
        """
        callback.Callback.__init__(self, **kwargs)

        self.show_cooldown = 1 / float(self._metadata["argv"].fps)

        self.frame_data = []
        self.frame_indices = []

        self.interactive_mode = self._metadata["argv"].interactive

        self.paused = False

        # canvas = FigureCanvas(self.fig)
        if mode == "square":
            self.width = int(len(animations) ** 0.5 + 0.5)
            self.height = self.width - (self.width**2 - len(animations)) // self.width
        if mode == "vstack":
            self.width = 1
            self.height = len(animations)
        elif mode == "hstack":
            self.height = 1
            self.width = len(animations)
        else:
            assert mode in ["square", "vstack", "hstack"]

        self.figsize = (self.height * 3, self.width * 3)

        if fig is None and self.interactive_mode:
            self.fig = PausableFigure(log=self.log, figsize=self.figsize)
            self.mng = backend_qt5agg.new_figure_manager_given_figure(1, self.fig)
            win = self.fig.canvas.window()
            self.fig.subplots_adjust(hspace=0.5, wspace=0.5)
            #win.setFixedSize(win.size())
        elif fig is None:
            self.fig = Figure(figsize=self.figsize)
            self.mng = None
        else:
            self.fig = fig
            self.mng = mng

        if ax is None:
            gs = GridSpec(self.width, self.height, figure=self.fig)
        else:
            gs = GridSpecFromSubplotSpec(self.width, self.height, subplot_spec=ax)

        self.animations = []
        for i, animation in enumerate(animations):
            self.animations.append(
                animation(
                    interactive=False, fig=self.fig, ax=gs[i], mng=self.mng, **kwargs
                )
            )

        # self.animations = [Animation(interactive=False, fig=self.fig, ax=gs, **kwargs) for Animation in animations]

        self.save_directory = Path(
            f".callbacks/{self.__class__.__name__}"
            + (f"@{self.attachee.__name__}" if self.attachee is not None else "")
        ).resolve()

        for animation in self.animations:
            animation.interactive_mode = self._metadata["argv"].interactive
            animation.setup()
        if self.interactive_mode:
            self.fig.show()

        self.ax = None
        self.last_show = time.time()
        # plt.show(block=True)

    @property
    def latest_frame(self):
        return [animation.latest_frame for animation in self.animations]

    @property
    def latest_frame_idx(self):
        return [animation.latest_frame_idx for animation in self.animations]

    def on_function_call(self, obj, method, output):
        try:
            self.frame_data.append(self.latest_frame)
            self.frame_indices.append(self.latest_frame_idx)
        except (IndexError, PrematureResolutionError):
            pass
        if self.interactive_mode:
            present_show = time.time()
            if present_show - self.last_show > self.show_cooldown:
                # self.fig.tight_layout()
                self.fig.canvas.draw()
                self.last_show = present_show
            self.fig.canvas.flush_events()
            while self.fig.paused:
                self.fig.canvas.flush_events()

    def construct_frame(self, *frame_data_of_animations):
        actor_bundles = []
        for animation, datum in zip(self.animations, frame_data_of_animations):
            actor_bundle = (
                animation.construct_frame(**datum)
                if not isinstance(animation, ComposedAnimationCallback)
                else animation.construct_frame(*datum)
            )
            actor_bundles.append(actor_bundle)
        return [actor for actors in actor_bundles for actor in actors]

    def setup(self):
        for animation in self.animations:
            animation.setup()

    def is_target_event(self, obj, method, output, triggers):
        return True # This is too much

    def on_launch(self):
        os.mkdir(self.get_save_directory())
        for animation in self.animations:
            animation.on_launch()

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        playback = (
            self._metadata["argv"].playback
            and iteration_number == iterations_total
            and episode_number == episodes_total
        )
        save_animation = self._metadata["argv"].save_animation
        if playback or save_animation:
            self.animate_and_save(
                name=f"ep{episode_number}|{episodes_total},it{iteration_number}|{iterations_total}"
            )
        for animation in self.animations:
            animation.on_episode_done(
                scenario,
                episode_number,
                episodes_total,
                iteration_number,
                iterations_total,
            )
        self.frame_data = []

    def on_iteration_done(self, *args, **kwargs):
        for animation in self.animations:
            animation.on_iteration_done(*args, **kwargs)

    def __del__(self):
        for animation in self.animations:
            del animation

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        for animation in self.animations:
            animation(*args, **kwargs)

    def lim(self, *args, frame_idx=None, **kwargs):
        for animation, idx in zip(self.animations, frame_idx):
            animation.lim(*args, frame_idx=idx, **kwargs)

    def animate(self, frames=None):
        plt.ioff()
        fig = Figure(figsize=self.figsize)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        canvas = FigureCanvas(fig)
        gs = GridSpec(self.width, self.height, figure=self.fig)
        self.install_temp_axis(fig, gs, interactive=False)
        if frames is None:
            frames = self.__class__._frames
        if frames == "all":
            frames = len(self.frame_data)
        elif isinstance(frames, int):
            frames = max(min(frames, len(self.frame_data)), 0)
        elif isinstance(frames, float) and 0 <= frames <= 1:
            frames = int(frames * len(self.frame_data))
        else:
            raise ValueError(
                'animate accepts an int, a float or "all", but instead a different value was provided.'
            )
        if not hasattr(self, "trajectory"):

            def animation_update(i):
                j = int((i / frames) * len(self.frame_data) + 0.5)
                self.lim(frame_idx=self.frame_indices[j])
                return self.construct_frame(*self.frame_data[j])

        else:
            trajectory_xs = self.trajectory_xs
            trajectory_ys = self.trajectory_ys

            def animation_update(i):
                j = int((i / frames) * len(self.frame_data) + 0.5)
                self.lim(frame_idx=j)
                return self.construct_frame(
                    trajectory_xs=trajectory_xs[:j],
                    trajectory_ys=trajectory_ys[:j],
                    *self.frame_data[j],
                )

        self.setup()

        anim = matplotlib.animation.FuncAnimation(
            self.fig,
            animation_update,
            frames=frames,
            interval=30,
            blit=True,
            repeat=False,
        )
        res = anim.to_jshtml()
        plt.close(self.fig)

        self.pop_axis()

        return res

    def install_temp_axis(self, fig, gs, interactive=None):
        if self.ax is not None:
            gs = GridSpecFromSubplotSpec(self.height, self.width, subplot_spec=gs)
        self._old_fig = self.fig
        self.fig = fig
        for i, animation in enumerate(self.animations):
            animation.install_temp_axis(fig, gs[i], interactive=interactive)

    def pop_axis(self):
        self.fig = self._old_fig
        del self._old_fig
        for animation in self.animations:
            animation.pop_axis()


class PrematureResolutionError(Exception):
    pass


class DeferredComposedAnimation(ComposedAnimationCallback, ABC):
    cooldown = None

    def __init__(
        self, *animations, fig=None, ax=None, mode="vstack", mng=None, **kwargs
    ):
        self.initialized = False
        if "interactive" in kwargs:
            del kwargs["interactive"]
        callback.Callback.__init__(self, **kwargs)

        """
        self.fig = Figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.fig)
        self._animation_classes = animations
        self._kwargs = kwargs
        """

        if fig is None:
            self.fig = Figure(figsize=(10, 10))
            self.mng = backend_qt5agg.new_figure_manager_given_figure(1, self.fig)
        else:
            self.fig = fig
            self.mng = mng

        self.ax = ax
        self.mode = mode
        self._animation_classes = animations
        # canvas = FigureCanvas(self.fig)

        self._kwargs = kwargs

        self.save_directory = Path(
            f".callbacks/{self.__class__.__name__}"
            + (f"@{self.attachee.__name__}" if self.attachee is not None else "")
        ).resolve()

    def on_launch(self):
        if self.initialized:
            super().on_launch()

    @property
    def animations(self):
        if self.initialized:
            return self._animations
        else:
            raise PrematureResolutionError(
                f"Tried to access attribute 'animation' of class {self.__class__.__name__} before the instance initialized."
            )

    def on_function_call(self, obj, method, output):
        pass

    def __deferred_init__(self):
        """
        self.animations = [
            Animation(interactive=False, **self._kwargs) for Animation in self._animation_classes
        ]
        width = int(len(self.animations) ** 0.5 + 0.5)
        height = width - (width**2 - len(self.animations)) // width
        for i, animation in enumerate(self.animations):
            animation.fig = self.fig
            animation.ax = self.canvas.figure.add_subplot(width, height, 1 + i)
        self.mng = backend_qt5agg.new_figure_manager_given_figure(1, self.fig)
        for animation in self.animations:
            animation.mng = self.mng
            animation.interactive_mode = self._metadata["argv"].interactive
            animation.setup()
            animation.fig.show()
        """

        if self.mode == "square":
            self.width = int(len(self._animation_classes) ** 0.5 + 0.5)
            self.height = (
                self.width
                - (self.width**2 - len(self._animation_classes)) // self.width
            )
        if self.mode == "vstack":
            self.width = 1
            self.height = len(self._animation_classes)
        elif self.mode == "hstack":
            self.height = 1
            self.width = len(self._animation_classes)
        else:
            assert self.mode in ["square", "vstack", "hstack"]

        if self.ax is None:
            self.gs = GridSpec(self.height, self.width, figure=self.fig)
        else:
            self.gs = GridSpecFromSubplotSpec(
                self.height, self.width, subplot_spec=self.ax
            )

        self._animations = []
        for i, animation in enumerate(self._animation_classes):
            self._animations.append(
                animation(
                    interactive=False,
                    fig=self.fig,
                    ax=self.gs[i],
                    mng=self.mng,
                    **self._kwargs,
                )
            )

        # self.animations = [Animation(interactive=False, fig=self.fig, ax=gs, **kwargs) for Animation in animations]

        for animation in self._animations:
            animation.interactive_mode = self._metadata["argv"].interactive
            animation.setup()
        # self.fig.show()

        self.initialized = True
        del self._animation_classes
        del self._kwargs

    def __call__(self, *args, **kwargs):
        if self.initialized:
            for animation in self.animations:
                animation(*args, **kwargs)
        else:
            callback.Callback.__call__(self, *args, **kwargs)

    def setup(self):
        if self.initialized:
            for animation in self.animations:
                animation.setup()

    def on_episode_done(self, *args, **kwargs):
        for animation in self.animations:
            animation.on_episode_done(*args, **kwargs)


class StateAnimation(DeferredComposedAnimation, callback.StateTracker):
    """A graph animation that plots values of state variables vs. time."""
    def __deferred_init__(self):
        state_dimension = len(self.system_state)
        self._animation_classes = []
        for i in range(state_dimension):

            def stateComponent(*args, component=i, **kwargs):
                return StateComponentAnimation(*args, component=component, last_component=(component == (state_dimension - 1)),
                                               fontsize=min(8.0, 15.0/state_dimension**0.50), **kwargs)

            self._animation_classes.append(stateComponent)
        super().__deferred_init__()

    def is_target_event(self, obj, method, output, triggers):
        return callback.StateTracker in triggers

    def on_trigger(self, _):
        self.__deferred_init__()
        super().on_launch()  # a bit clumsy, I think


class ObservationAnimation(DeferredComposedAnimation, callback.ObservationTracker):
    """A graph animation that plots observables vs. time."""
    def __deferred_init__(self):
        observation_dimension = len(self.observation)
        self._animation_classes = []
        for i in range(observation_dimension):

            def observationComponent(*args, component=i, **kwargs):
                return ObservationComponentAnimation(
                    *args, component=component, fontsize=min(10.0, 15.0/observation_dimension**0.5), **kwargs
                )

            self._animation_classes.append(observationComponent)
        super().__deferred_init__()

    def is_target_event(self, obj, method, output, triggers):
        return callback.ObservationTracker in triggers

    def on_trigger(self, _):
        self.__deferred_init__()
        super().on_launch()  # a bit clumsy, I think


class ActionAnimation(DeferredComposedAnimation, callback.ActionTracker):
    """A graph animation that plots control inputs vs. time."""
    def __deferred_init__(self):
        action_dimension = len(self.action)
        self._animation_classes = []
        for i in range(action_dimension):

            def actionComponent(*args, component=i, **kwargs):
                return ActionComponentAnimation(*args, component=component, last_component=(component == action_dimension - 1),
                                                fontsize=min(8.0, 15.0/action_dimension**0.5), **kwargs)

            self._animation_classes.append(actionComponent)
        super().__deferred_init__()

    def is_target_event(self, obj, method, output, triggers):
        return callback.ActionTracker in triggers

    def on_trigger(self, _):
        self.__deferred_init__()
        super().on_launch()  # a bit clumsy, I think


class ObjectiveAnimation(DeferredComposedAnimation, callback.ObjectiveTracker):
    def __deferred_init__(self):
        objective_dimension = len(self.objective)
        self._animation_classes = []
        for i in range(objective_dimension):

            def objectiveComponent(*args, component=i, **kwargs):
                return ObjectiveComponentAnimation(*args, component=component, fontsize=min(10.0, 15.0/objective_dimension**0.5), **kwargs)

            self._animation_classes.append(objectiveComponent)
        super().__deferred_init__()

    def is_target_event(self, obj, method, output, triggers):
        return callback.ObjectiveTracker in triggers

    def on_trigger(self, _):
        self.__deferred_init__()
        super().on_launch()


class GraphAnimation(AnimationCallback):
    """Animation of graphs that adds more data to curves over time."""
    _legend = (None,)
    _vertices = 4000
    _line = "-"

    def add_frame(self, line_datas):
        line_datas = list(line_datas)
        for i in range(len(line_datas)):
            t, y = line_datas[i]
            line_datas[i] = (list(t), list(y))
        super().add_frame(line_datas=line_datas)

    def setup(self):
        super().setup()
        self.lines = []
        for name in self._legend:
            (line,) = self.ax.plot([0], [0], self._line, label=name)
            self.lines.append(line)

    def lim(self, *args, extra_margin=0.01, **kwargs):
        datum = self.frame_data[-1]["line_datas"]
        ts, ys = [], []
        for line_data in datum:
            t, y = line_data
            skip = 1 + len(t) // self._vertices
            ts += list(t)[::skip]
            ys += list(y)[::skip]

        left, right, bottom, top = self.lim_from_reference(
            np.array(ts), np.array(ys), extra_margin, equal=False
        )
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)

    def construct_frame(self, line_datas):
        for line, data in zip(self.lines, line_datas):
            t, y = data
            skip = 1 + len(t) // self._vertices
            downsampling = slice(1, None, skip) if isinstance(self, callback.TimeTracker) else slice(None, None, skip)
            line.set_data(t[downsampling], y[downsampling]) # these slices are to avoid residual time bug
        return self.lines


class ValueAnimation(GraphAnimation, callback.ValueTracker):
    """A graph animation that displays that plots episode-mean value over iterations."""
    _legend = (None,)
    _is_global = True

    def setup(self):
        super().setup()
        if not hasattr(self, "scores"):
            self.t = []
            self.y = []
            self.scores = []
            self.ax.set_axis_off()
        self.ax.set_xlabel("Learning Iteration")
        self.ax.set_ylabel("Value")
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.add_frame(line_datas=[(self.t, self.y)])

    def install_temp_axis(self, *args, **kwargs):
        super().install_temp_axis(*args, **kwargs)
        self.lines_old = self.lines

    def pop_axis(self):
        super().pop_axis()
        self.lines = self.lines_old
        del self.lines_old

    def is_target_event(self, obj, method, output, triggers):
        return callback.ValueTracker in triggers

    def on_trigger(self, _):
        self.scores.append(self.score)

    def on_iteration_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        if iteration_number >= 2:
            self.ax.set_axis_on()
        if self.scores:
            self.t.append(iteration_number)
            self.y.append(np.mean(self.scores))
            self.add_frame(
                line_datas=[(self.t, self.y)]
            )  # these slices are there to avoid residual time bug
        self.scores = []

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        pass #self.setup()


class StateComponentAnimation(
    GraphAnimation, callback.StateTracker, callback.TimeTracker
):
    _legend = (None,)

    def __init__(self, *args, component=0, fontsize=10.0, last_component=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.component = component
        self.last_component = last_component
        self.fontsize = fontsize

    def get_save_directory(self):
        return super().get_save_directory() + str(self.component)

    def setup(self):
        super().setup()
        #self.ax.tick_params(axis='y', labelrotation=45)
        self.ax.tick_params(axis='y',  labelsize=self.fontsize)
        if self.last_component:
            self.ax.set_xlabel("Time [s]")
        if hasattr(self, "label_of_y"):
            self.ax.set_ylabel(self.label_of_y, fontsize=self.fontsize, rotation=80)
        self.t = []
        self.y = []

    def is_target_event(self, obj, method, output, triggers):
        return callback.StateTracker in triggers

    def on_trigger(self, _):
        if not hasattr(self, "label_of_y"):
            self.label_of_y = title_except_units(self.state_naming[self.component])
            self.ax.set_ylabel(self.label_of_y, fontsize=self.fontsize, rotation=80)
        self.t.append(self.time)
        self.y.append(self.system_state[self.component])
        self.add_frame(
            line_datas=[(self.t, self.y)]
        )  # these slices are there to avoid residual time bug


class ObservationComponentAnimation(  # TO DO: introduce an abstract class ObservablesAnimation which will include its own ObservableComponentAnimation.
    GraphAnimation, callback.ObservationTracker, callback.TimeTracker
):
    _legend = (None,)

    def __init__(self, *args, component=0, fontsize=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.component = component
        self.fontsize = fontsize

    def get_save_directory(self):
        return super().get_save_directory() + str(self.component)

    def setup(self):
        super().setup()
        self.ax.set_xlabel("Time [s]")
        if hasattr(self, "label_of_y"):
            self.ax.set_ylabel(self.label_of_y, fontsize=self.fontsize)
        self.t = []
        self.y = []

    def is_target_event(self, obj, method, output, triggers):
        return callback.ObservationTracker in triggers

    def on_trigger(self, _):
        if not hasattr(self, "label_of_y"):
            self.label_of_y = title_except_units(self.observation_naming[self.component])
            self.ax.set_ylabel(self.label_of_y, fontsize=self.fontsize)
        self.t.append(self.time)
        self.y.append(self.observation[self.component])
        self.add_frame(
            line_datas=[(self.t, self.y)]
        )  # these slices are there to avoid residual time bug


class ActionComponentAnimation(  # TO DO: introduce an abstract class ObservablesAnimation which will include its own ObservableComponentAnimation.
    GraphAnimation, callback.ActionTracker, callback.TimeTracker
):
    _legend = (None,)
    _line = "g-"

    def __init__(self, *args, component=0, last_component=False, fontsize=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.component = component
        self.fontsize = fontsize
        self.last_component = last_component

    def get_save_directory(self):
        return super().get_save_directory() + str(self.component)

    def setup(self):
        super().setup()
        self.ax.tick_params(axis='y', labelsize=self.fontsize)
        if self.last_component:
            self.ax.set_xlabel("Time [s]")
        if hasattr(self, "label_of_y"):
            self.ax.set_ylabel(self.label_of_y, fontsize=self.fontsize, rotation=80)
        self.t = []
        self.y = []

    def is_target_event(self, obj, method, output, triggers):
        return callback.ActionTracker in triggers

    def on_trigger(self, _):
        if not hasattr(self, "label_of_y"):
            self.label_of_y = title_except_units(self.action_naming[self.component])
            self.ax.set_ylabel(self.label_of_y, fontsize=self.fontsize, rotation=80)
        self.t.append(self.time)
        self.y.append(self.action[self.component])
        self.add_frame(
            line_datas=[(self.t, self.y)]
        )  # these slices are there to avoid residual time bug


class ObjectiveComponentAnimation(  # TO DO: introduce an abstract class ObservablesAnimation which will include its own ObservableComponentAnimation.
    GraphAnimation, callback.ObjectiveTracker, callback.TimeTracker
):
    _legend = (None,)
    _line = "r-"

    def __init__(self, *args, component=0, fontsize=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.component = component
        self.fontsize = fontsize

    def get_save_directory(self):
        return super().get_save_directory() + str(self.component)

    def setup(self):
        super().setup()
        self.ax.set_xlabel("Time [s]")
        if hasattr(self, "label_of_y"):
            self.ax.set_ylabel(self.label_of_y, fontsize=self.fontsize)
        self.t = []
        self.y = []

    def is_target_event(self, obj, method, output, triggers):
        return callback.ObjectiveTracker in triggers

    def on_trigger(self, _):
        if not hasattr(self, "label_of_y"):
            self.label_of_y = title_except_units(self.objective_naming[self.component])
            self.ax.set_ylabel(self.label_of_y, fontsize=self.fontsize)
        self.t.append(self.time)
        self.y.append(self.objective[self.component])
        self.add_frame(
            line_datas=[(self.t, self.y)]
        )  # these slices are there to avoid residual time bug


class MultiSpriteAnimation(AnimationCallback, ABC):
    """Animation that sets the location and rotation of a planar shape at each frame."""

    _pics = None  # must be svgs located in regelum/img
    _marker_sizes = 30
    _rotations = 0
    _center_vert = False

    def setup(self):
        super().setup()
        self.trajectory_xs = []
        self.trajectory_ys = []
        (self.trajectory,) = self.ax.plot(self.trajectory_xs, self.trajectory_ys, "--")
        self.paths = [
            regelum.__file__.replace("__init__.py", f"img/{pic}") for pic in self._pics
        ]
        self.markers = []
        self.sprites = []
        self.pic_datas = []
        self.attributes = []
        self.original_transforms = []
        if not isinstance(self._rotations, Iterable):
            rotations = [self._rotations] * len(self.paths)
        else:
            rotations = self._rotations
        if not isinstance(self._marker_sizes, Iterable):
            sizes = [self._marker_sizes] * len(self.paths)
        else:
            sizes = self._marker_sizes
        for path, rotation, size in zip(self.paths, rotations, sizes):
            pic_data, attribute = svg2paths(path)
            parsed = parse_path(attribute[0]["d"])
            parsed.vertices[:, 0] -= parsed.vertices[:, 0].mean(axis=0)
            if self._center_vert:
                parsed.vertices[:, 1] -= parsed.vertices[:, 1].mean(axis=0)
            marker = matplotlib.markers.MarkerStyle(marker=parsed)
            marker._transform = marker.get_transform().rotate_deg(rotation)
            (sprite,) = self.ax.plot(0, 1, marker=marker, ms=size)
            original_transform = marker.get_transform()
            self.markers.append(marker)
            self.sprites.append(sprite)
            self.pic_datas.append(pic_data)
            self.attributes.append(attribute)
            self.original_transforms.append(original_transform)

    def increment_trajectory(self, x0=None, y0=None, **frame_datum):
        return x0, y0

    def lim(self, *args, extra_margin=0.11, **kwargs):
        x, y = np.array([list(datum.values()) for datum in self.frame_data]).T[:2]
        left, right, bottom, top = self.lim_from_reference(x, y, extra_margin)
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)

    def add_frame(self, **frame_datum):
        traj_x, traj_y = self.increment_trajectory(**frame_datum)
        self.trajectory_xs.append(traj_x)
        self.trajectory_ys.append(traj_y)
        frame_datum["trajectory_data"] = (
            list(self.trajectory_xs),
            list(self.trajectory_ys),
        )
        return super().add_frame(**frame_datum)

    def construct_frame(self, trajectory_data=None, **kwargs):
        self.trajectory.set_data(*trajectory_data)
        for i in range(len(self.paths)):
            x, y, theta = kwargs[f"x{i}"], kwargs[f"y{i}"], kwargs[f"theta{i}"]
            self.sprites[i].set_data([x], [y])
            self.markers[i]._transform = Affine2D(
                self.original_transforms[i]._mtx.copy()
            )
            self.markers[i]._transform = (
                self.markers[i].get_transform().rotate_deg(180 * theta / np.pi)
            )
        return self.sprites


class PlanarBodiesAnimation(MultiSpriteAnimation, callback.StateTracker):
    """Animates dynamics of systems that can be viewed as a set of bodies moving on a plane."""

    def setup(self):
        super().setup()
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        # self.ax.set_title(f"Planar motion of {self.attachee.__name__}")

    def on_trigger(self, _):
        self.add_frame(
            x0=self.system_state[0],
            y0=self.system_state[1],
            theta0=self.system_state[2],
        )


class CartpoleAnimation(PlanarBodiesAnimation):
    """Animates dynamics of Lunar Lander that can be viewed as a triangle moving on a plane."""

    _pics = ["cart.svg", "pendulum.svg"]
    _marker_sizes = [33, 55]

    def lim(self, *args, frame_idx=None, extra_margin=0.11, **kwargs):
        x = self.frame_data[frame_idx]["x0"]
        left, right, bottom, top = self.lim_from_reference(
            np.array([x - 2, x + 2]), np.array([-1, 3]), extra_margin, equal=True
        )
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)

    def increment_trajectory(self, x0, theta1, **kwargs):
        return x0 + np.cos(theta1 + np.pi / 2), np.sin(theta1 + np.pi / 2)

    def setup(self):
        super().setup()
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("")
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_aspect("equal", adjustable="box")
        (self.ground,) = self.ax.plot([-100000, 100000], [0, 0], "r", ms=10, zorder=-1)
        (self.target,) = self.ax.plot([0], [0], "g", ms=35, marker="o", zorder=-1)
        self.target_text = self.ax.text(
            0,
            0,
            "Target",
            horizontalalignment="center",
            verticalalignment="center",
            zorder=-1,
        ).set_clip_on(True)

    def on_trigger(self, _):
        self.add_frame(
            x0=self.system_state[1],
            y0=0,
            theta0=0,
            x1=self.system_state[1],
            y1=0,
            theta1=self.system_state[0],
        )


class SingleBodyAnimation(PlanarBodiesAnimation):
    """Animates dynamics of systems that can be viewed as a body moving on a plane."""

    _pic = "default.svg"
    _marker_size = None
    _rotation = None

    def setup(self):
        self._pics = [self._pic]
        if self._marker_size is not None:
            self._marker_sizes = [self._marker_size]
        if self._rotation is not None:
            self._rotations = [self._rotation]
        super().setup()

    def add_frame(self, **frame_datum):
        if "x" in frame_datum:
            frame_datum["x0"] = frame_datum["x"]
        if "y" in frame_datum:
            frame_datum["y0"] = frame_datum["y"]
        if "theta" in frame_datum:
            frame_datum["theta0"] = frame_datum["theta"]
        super().add_frame(**frame_datum)


class LunarLanderAnimation(SingleBodyAnimation):
    """Animates dynamics of Lunar Lander that can be viewed as a triangle moving on a plane."""

    _pic = "lunar_lander.svg"
    _marker_size = 30
    _center_vert = True

    def lim(self, *args, frame_idx=None, extra_margin=0.11, **kwargs):
        self.sprites[0].set_visible(True)

        x = self.frame_data[frame_idx]["x"]
        left, right, bottom, top = self.lim_from_reference(
            np.array([x - 2, x + 2]), np.array([0, 5]), extra_margin, equal=False
        )
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)

    def setup(self):
        super().setup()
        #(self.ground,) = self.ax.plot([-100000, 100000], [0, 0], "r", ms=10)
        self.ax.set_ylim(0 - 0.11, 5 + 0.11)
        self.ax.set_xlim(-2 - 0.11, 2 + 0.11)
        self.sprites[0].set_visible(False)
        rect = Rectangle((-10000, -1), 20000, 1, linewidth=1, edgecolor='grey', facecolor='grey')
        self.ground = self.ax.add_patch(rect)
        (self.target,) = self.ax.plot([0], [0], "g", ms=35, marker="o")
        self.target_text = self.ax.text(
            0, 0, "Target", horizontalalignment="center", verticalalignment="center"
        ).set_clip_on(True)

    def on_trigger(self, _):
        self.add_frame(
            x=self.system_state[2], y=self.system_state[1], theta=self.system_state[0]
        )


class OmnirobotAnimation(SingleBodyAnimation):
    _pic = "omnirobot.svg"
    _marker_size = 30
    _center_vert = True

    def on_trigger(self, _):
        self.add_frame(x=self.system_state[0], y=self.system_state[1], theta=0)

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)


class PendulumAnimation(SingleBodyAnimation):
    """Animates the head of a swinging pendulum.

    Interprets the first state coordinate as the angle of the pendulum with respect to the topmost position.
    """

    _pic = "pendulum.svg"
    _marker_size = 180

    def setup(self):
        super().setup()
        self.ax.get_yaxis().set_visible(False)
        self.ax.get_xaxis().set_visible(False)
        self.ax.set_aspect("equal", adjustable="box")
        (self.trajectory,) = self.ax.plot(
            self.trajectory_xs, self.trajectory_ys, "--", alpha=0.6
        )

    def on_trigger(self, _):
        self.add_frame(x=0, y=0, theta=self.system_state[0])

    def increment_trajectory(self, x, y, theta, **kwargs):
        return np.cos(theta + np.pi / 2) * 0.9, np.sin(theta + np.pi / 2) * 0.9

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)


class ThreeWheeledRobotAnimation(SingleBodyAnimation):
    """Animates the position of a differential platform."""

    _pic = "3wrobot.svg"
    _rotation = 225
    # _marker_size = 250
    # _frames = 500

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-7, 7)
        self.ax.set_ylim(-7, 7)


class BarAnimation(AnimationCallback, callback.StateTracker):
    """Animates the state of a system as a collection of vertical bars."""

    _naming = None

    def setup(self):
        super().setup()
        self.bar = None

    def construct_frame(self, state):
        if self.bar is None:
            self.bar = self.ax.bar(self.state_naming, state)
        else:
            for rect, h in zip(self.bar, state):
                rect.set_height(h)
        return self.bar

    def on_trigger(self, _):
        self.add_frame(state=self.system_state)

    def lim(self, *args, **kwargs):
        pass


@contextmanager
def preserve_limits(ax=None):
    """ Plot without modifying axis limits """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    try:
        yield ax
    finally:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

class TwoTankAnimation(BarAnimation):
    def setup(self):
        super().setup()
        self.level = self.ax.plot([-10000, 10000], [0.44, 0.44], 'r--')
        pump = Rectangle((0.4, 0), 0.35, 0.1, zorder=10, facecolor="grey")
        self.ax.add_patch(pump)
        self.ax.text(0.4, 0.01, "Pump", fontsize=8, zorder=11)

        sink = Rectangle((1.2, 0), 0.4, 0.1, zorder=10, facecolor="grey")
        self.ax.add_patch(sink)
        self.ax.text(1.2, 0.01, "Sink", fontsize=8, zorder=11)

        intake = Rectangle((-0.5, 1.8), 0.4, 0.1, zorder=10, facecolor="grey")
        self.ax.add_patch(intake)
        self.ax.text(-0.5, 1.81, "Intake", fontsize=8, zorder=11)

        self.bar = self.ax.bar(["", " "], [0, 0])

        self.ax.set_xlabel("Intake/Sink Level [m]")

        self.ax.set_xlim(-0.5, 1.5)



DefaultAnimation = StateAnimation + ActionAnimation + ValueAnimation
