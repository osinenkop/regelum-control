#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains an interface class `animator` along with concrete realizations, each of which is associated with a corresponding system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from ..__utilities import rc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage

# !pip install mpldatacursor <-- to install this
from mpldatacursor import datacursor

# !pip install svgpath2mpl matplotlib <-- to install this
from svgpath2mpl import parse_path
import numbers
from abc import ABC, abstractmethod
from itertools import product


def update_line(matplotlib_handle, newX, newY):
    old_xdata = matplotlib_handle.get_xdata()
    old_ydata = matplotlib_handle.get_ydata()
    if any(isinstance(coord, numbers.Number) for coord in [newX, newY]):
        new_xdata = rc.append(old_xdata, newX)
        new_ydata = rc.append(old_ydata, newY)
    else:
        new_xdata = rc.concatenate((old_xdata, newX))
        new_ydata = rc.concatenate((old_ydata, newY))

    matplotlib_handle.set_xdata(new_xdata)
    matplotlib_handle.set_ydata(new_ydata)


def reset_line(matplotlib_handle):
    matplotlib_handle.set_data([], [])


def update_text(matplotlib_handle, new_text):
    matplotlib_handle.set_text(new_text)


def update_patch_color(matplotlib_handle, new_color):
    matplotlib_handle.set_color(str(new_color))


def init_data_cursor(matplotlib_handle):
    datacursor(matplotlib_handle)


class Dashboard(ABC):
    def __init__(self):
        self.artists = []

    @abstractmethod
    def init_dashboard(self):
        pass

    def perform_step_update(self):
        pass
        # raise NotImplementedError(
        #     f"step update method is not implemented for dashboard {self.__class__.__name__}"
        # )

    def perform_episodic_update(self):
        pass
        # raise NotImplementedError(
        #     f"episodic update not implemented for dashboard {self.__class__.__name__}"
        # )

    def perform_iterative_update(self):
        pass
        # raise NotImplementedError(
        #     f"iterative update method not implemented for dashboard {self.__class__.__name__}"
        # )

    def update(self, update_variant):
        if update_variant == "step":
            self.perform_step_update()
        elif update_variant == "episode":
            self.perform_episodic_update()
        elif update_variant == "iteration":
            self.perform_iterative_update()


class Animator:
    """
    Interface class of visualization machinery for simulation of system-controller loops.
    To design a concrete animator: inherit this class, override:
        | :func:`~animators.Animator.__init__` :
        | define necessary visual elements (required)
        | :func:`~animators.Animator.init_anim` :
        | initialize necessary visual elements (required)
        | :func:`~animators.Animator.animate` :
        | animate visual elements (required)

    Attributes
    ----------
    objects : : tuple
        Objects to be updated within animation cycle
    pars : : tuple
        Fixed parameters of objects and visual elements

    """

    def __init__(self, subplot_grid_size=[2, 2]):
        self.subplot_grid_size = subplot_grid_size
        self.artists = []

    def init_anim(self):
        self.main_figure = plt.figure(figsize=(10, 10))
        self.axes_array = self.main_figure.subplots(*self.subplot_grid_size)
        for r, c in product(
            range(self.subplot_grid_size[0]), range(self.subplot_grid_size[1])
        ):
            plt.sca(self.axes_array[r, c])  ####---Set current axes
            self.dashboards[self.get_index(r, c)].init_dashboard()

        return self.artists

    def update_dashboards(self, update_variant):
        for r, c in product(
            range(self.subplot_grid_size[0]), range(self.subplot_grid_size[1])
        ):
            plt.sca(self.axes_array[r, c])
            self.dashboards[self.get_index(r, c)].update(update_variant)

        return self.artists

    def animate(self, k):
        sim_status = self.scenario.step()
        SIMULATION_ENDED = sim_status == "simulation_ended"
        EPISODE_ENDED = sim_status == "episode_ended"
        ITERATION_ENDED = sim_status == "iteration_ended"

        if SIMULATION_ENDED:
            print("Simulation ended")
            self.anm.event_source.stop()

        elif EPISODE_ENDED:
            self.update_dashboards("episode")
        elif ITERATION_ENDED:
            self.update_dashboards("iteration")
        else:
            self.update_dashboards("step")

        return self.artists

    def get_anm(self, anm):
        """
        ``anm`` should be a ``FuncAnimation`` object.
        This method is needed to hand the animator access to the currently running animation, say, via ``anm.event_source.stop()``.

        """
        self.anm = anm

    def stop_anm(self):
        """
        Stops animation, provided that ``self.anm`` was defined via ``get_anm``.

        """
        self.anm.event_source.stop()
        # plt.close('all')
        raise Exception("exit")

    def set_sim_data(self, **kwargs):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        """
        self.__dict__.update(kwargs)

    def collect_dashboards(self, *dashboards):
        self.dashboards = dashboards
        for dashboard in self.dashboards:
            self.artists.extend(dashboard.artists)

    def get_index(self, r, c):
        return r * self.subplot_grid_size[1] + c

    def playback(self):
        from matplotlib import animation
        from rcognita.__utilities import on_key_press

        self.init_anim()

        anm = animation.FuncAnimation(
            self.main_figure,
            self.animate,
            blit=True,
            interval=self.sampling_time / 1e6,
            repeat=False,
        )

        self.get_anm(anm)

        cId = self.main_figure.canvas.mpl_connect(
            "key_press_event", lambda event: on_key_press(event, anm)
        )

        anm.running = True


class RobotMarker:
    """
    Robot marker for visualization.

    """

    def __init__(self, angle=None, path_string=None):
        self.angle = angle or []
        self.path_string = (
            path_string
            or """m 66.893258,227.10128 h 5.37899 v 0.91881 h 1.65571 l 1e-5,-3.8513 3.68556,-1e-5 v -1.43933
        l -2.23863,10e-6 v -2.73937 l 5.379,-1e-5 v 2.73938 h -2.23862 v 1.43933 h 3.68556 v 8.60486 l -3.68556,1e-5 v 1.43158
        h 2.23862 v 2.73989 h -5.37899 l -1e-5,-2.73989 h 2.23863 v -1.43159 h -3.68556 v -3.8513 h -1.65573 l 1e-5,0.91881 h -5.379 z"""
        )
        self.path = parse_path(self.path_string)
        self.path.vertices -= self.path.vertices.mean(axis=0)
        self.marker = matplotlib.markers.MarkerStyle(marker=self.path)
        self.marker._transform = self.marker.get_transform().rotate_deg(angle)

    def rotate(self, angle=0):
        self.marker._transform = self.marker.get_transform().rotate_deg(
            angle - self.angle
        )
        self.angle = angle

