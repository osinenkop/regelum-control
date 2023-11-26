"""Callbacks that create, display and store animation according to dynamic data."""

from abc import ABC, abstractmethod
from copy import copy
from unittest.mock import Mock

import matplotlib.animation
import matplotx.styles
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

plt.style.use(matplotx.styles.dracula)


class AnimationCallback(callback.Callback, ABC):
    """Callback (base) responsible for rendering animated visualizations of the experiment."""

    _frames = 100

    @classmethod
    def _compose(cls_left, cls_right):
        animations = []
        for cls in [cls_left, cls_right]:
            if issubclass(cls, ComposedAnimationCallback):
                assert (
                    cls is not ComposedAnimationCallback
                ), "Adding a composed animation in this way is ambiguous."
                animations += cls._animations
            else:
                animations.append(cls)

        class Composed(ComposedAnimationCallback):
            _animations = animations

            def __init__(self, **kwargs):
                super().__init__(*self._animations, **kwargs)

        return Composed

    def __init__(self, *args, interactive=None, **kwargs):
        """Initialize an instance of AnimationCallback."""
        super().__init__(*args, **kwargs)
        assert (
            self.attachee is not None
        ), "An animation callback can only be instantiated via attachment, however an attempt to instantiate it otherwise was made."
        self.frame_data = []
        # matplotlib.use('Qt5Agg', force=True)

        self.interactive_mode = (
            self._metadata["argv"].interactive if interactive is None else interactive
        )
        if self.interactive_mode:
            self.fig = Figure(figsize=(10, 10))
            canvas = FigureCanvas(self.fig)
            self.ax = canvas.figure.add_subplot(111)
            self.mng = backend_qt5agg.new_figure_manager_given_figure(1, self.fig)
            self.setup()
        else:
            self.mng = None
            self.fig, self.ax = None, None
        self.save_directory = Path(
            f".callbacks/{self.__class__.__name__}@{self.attachee.__name__}"
        ).resolve()
        self.saved_counter = 0

    @abstractmethod
    def setup(self):
        pass

    def get_save_directory(self):
        return str(self.save_directory)

    def add_frame(self, **frame_datum):
        self.frame_data.append(frame_datum)
        if hasattr(self, "interactive_status"):
            self.interactive_status["frame"] = frame_datum

        if self.interactive_mode:
            self.lim()
            self.construct_frame(**frame_datum)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.fig.show()

    def __getstate__(self):
        state = copy(self.__dict__)
        del state["mng"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mng = Mock()

    def reset(self):
        self.frame_data = []
        if self.interactive_mode:
            self.mng.window.close()
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
        self.setup()
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

        def animation_update(i):
            j = int((i / frames) * len(self.frame_data) + 0.5)
            return self.construct_frame(**self.frame_data[j])

        self.lim()
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
    def lim_from_reference(cls, x, y, extra_margin=0):
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1
        y_min, y_max = y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1
        left = x_min - extra_margin
        right = x_min + max(x_max - x_min, y_max - y_min) + extra_margin
        top = y_min - extra_margin
        bottom = y_min + max(x_max - x_min, y_max - y_min) + extra_margin
        return left, right, bottom, top

    def lim(self, width=None, height=None, center=None, extra_margin=0.0):
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
        self.animate_and_save(name=str(episode_number))
        self.reset()


class ComposedAnimationCallback(AnimationCallback):
    """An animation callback capable of incoroporating several other animation callbacks in such a way that the respective plots are distributed between axes of a single figure."""

    def __init__(self, *animations, **kwargs):
        """Initialize an instance of ComposedAnimationCallback.

        Args:
            *animations (*List[AnimationCallback]): animation classes to be composed
            **kwargs (**Dict[str, Any]): keyword arguments to be passed to __init__ of said
                animations and the base class
        """
        callback.Callback.__init__(self, **kwargs)
        self.animations = [
            Animation(interactive=False, **kwargs) for Animation in animations
        ]
        self.fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(self.fig)
        width = int(len(self.animations) ** 0.5 + 0.5)
        height = width - (width**2 - len(self.animations)) // width
        for i, animation in enumerate(self.animations):
            animation.fig = self.fig
            animation.ax = canvas.figure.add_subplot(width, height, 1 + i)
            animation.ax.set_aspect("equal", "box")
        self.mng = backend_qt5agg.new_figure_manager_given_figure(1, self.fig)
        for animation in self.animations:
            animation.mng = self.mng
            animation.interactive_mode = self._metadata["argv"].interactive
            animation.setup()

    def perform(self, obj, method, output):
        raise AssertionError(
            f"A {self.__class__.__name__} object is not meant to have its `perform` method called."
        )

    def construct_frame(self, **frame_datum):
        raise AssertionError(
            f"A {self.__class__.__name__} object is not meant to have its `construct_frame` method called."
        )

    def setup(self):
        raise AssertionError(
            f"A {self.__class__.__name__} object is not meant to have its `setup` method called."
        )

    def is_target_event(self, obj, method, output):
        raise AssertionError(
            f"A {self.__class__.__name__} object is not meant to have its `is_target_event` method called."
        )

    def on_launch(self):
        for animation in self.animations:
            animation.on_launch()

    def on_episode_done(self, *args, **kwargs):
        for animation in self.animations:
            animation.on_episode_done(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        for animation in self.animations:
            animation(*args, **kwargs)


class PointAnimation(AnimationCallback, ABC):
    """Animation that sets the location of a planar point at each frame."""

    def setup(self):
        (self.point,) = self.ax.plot(0, 1, marker="o", label="location")

    def construct_frame(self, x, y):
        self.point.set_data([x], [y])
        return (self.point,)

    def lim(self, *args, extra_margin=0.01, **kwargs):
        x, y = np.array([list(datum.values()) for datum in self.frame_data]).T[:2]
        left, right, bottom, top = self.lim_from_reference(x, y, extra_margin)
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)


class PlanarMotionAnimation(PointAnimation, callback.StateTracker):
    """Animates dynamics of systems that can be viewed as a point moving on a plane."""

    def on_trigger(self, _):
        self.add_frame(x=self.system_state[0], y=self.system_state[1])


class TriangleAnimation(AnimationCallback, ABC):
    """Animation that sets the location and rotation of a planar equilateral triangle at each frame."""

    _pic = None  # must be an svg located in regelum/img
    _ms = 30
    _rot = 0

    def setup(self):
        if self._pic is None:
            return self.setup_points()
        else:
            return self.setup_pic()

    def setup_points(self):
        (point1,) = self.ax.plot(0, 1, marker="o", label="location", ms=30)
        (point2,) = self.ax.plot(0, 1, marker="o", label="location", ms=30)
        (point3,) = self.ax.plot(0, 1, marker="o", label="location", ms=30)
        self.points = (point1, point2, point3)
        self.ax.grid()

    def setup_pic(self):
        self.path = regelum.__file__.replace("__init__.py", f"img/{self._pic}")
        self.pic_data, self.attributes = svg2paths(self.path)
        parsed = parse_path(self.attributes[0]["d"])
        parsed.vertices[:, 0] -= parsed.vertices[:, 0].mean(axis=0)
        self.marker = matplotlib.markers.MarkerStyle(marker=parsed)
        self.marker._transform = self.marker.get_transform().rotate_deg(self._rot)
        (self.triangle,) = self.ax.plot(0, 1, marker=self.marker, ms=self._ms)
        self.original_transform = self.marker.get_transform()

    def lim(self, *args, extra_margin=0.11, **kwargs):
        x, y = np.array([list(datum.values()) for datum in self.frame_data]).T[:2]
        left, right, bottom, top = self.lim_from_reference(x, y, extra_margin)
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)

    def construct_frame(self, x, y, theta):
        if self._pic is None:
            return self.construct_frame_points(x, y, theta)
        else:
            return self.construct_frame_pic(x, y, theta)

    def construct_frame_points(self, x, y, theta):
        offsets = (
            np.array(
                [
                    [
                        np.cos(theta + i * 2 * np.pi / 3),
                        np.sin(theta + i * 2 * np.pi / 3),
                    ]
                    for i in range(3)
                ]
            )
            / 10
        )
        location = np.array([x, y])
        for point, offset in zip(self.points, offsets):
            x, y = location + offset
            point.set_data([x], [y])
        return self.points

    def construct_frame_pic(self, x, y, theta):
        self.triangle.set_data([x], [y])
        self.marker._transform = Affine2D(self.original_transform._mtx.copy())
        self.marker._transform = self.marker.get_transform().rotate_deg(
            180 * theta / np.pi
        )
        return (self.triangle,)


class DirectionalPlanarMotionAnimation(TriangleAnimation, callback.StateTracker):
    """Animates dynamics of systems that can be viewed as a triangle moving on a plane."""

    def setup(self):
        super().setup()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title(f"Planar motion of {self.attachee.__name__}")

    def on_trigger(self, _):
        self.add_frame(
            x=self.system_state[0], y=self.system_state[1], theta=self.system_state[2]
        )


class DirectionalPlanarMotionAnimationLander(DirectionalPlanarMotionAnimation):
    """Animates dynamics of Lunar Lander that can be viewed as a triangle moving on a plane."""

    def on_trigger(self, _):
        self.add_frame(
            x=self.system_state[1], y=self.system_state[2], theta=self.system_state[0]
        )


class PendulumAnimation(DirectionalPlanarMotionAnimation):
    """Animates the head of a swinging pendulum.

    Interprets the first state coordinate as the angle of the pendulum with respect to the topmost position.
    """

    _pic = "pendulum.svg"
    _ms = 250

    def on_trigger(self, _):
        self.add_frame(x=0, y=0, theta=self.system_state[0])

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)


class ThreeWheeledRobotAnimation(DirectionalPlanarMotionAnimation):
    """Animates the position of a differential platform."""

    _pic = "3wrobot.svg"
    _rot = 225

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)


class BarAnimation(AnimationCallback, callback.StateTracker):
    """Animates the state of a system as a collection of vertical bars."""

    def setup(self):
        self.bar = None

    def construct_frame(self, state):
        if self.bar is None:
            self.bar = self.ax.bar(range(len(state)), state)
        else:
            for rect, h in zip(self.bar, state):
                rect.set_height(h)
        return self.bar

    def on_trigger(self, _):
        self.add_frame(state=self.system_state)

    def lim(self):
        pass
