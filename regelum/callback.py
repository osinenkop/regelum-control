"""Contains callbacks.

Callbacks are lightweight event handlers, used mainly for logging.

To make a method of a class trigger callbacks, one needs to
decorate the class (or its base class) with ``@introduce_callbacks()`` and then
also decorate the method with ``@apply_callbacks``.

Callbacks can be registered by simply supplying them in the respective keyword argument:
::

    @regelum.main(callbacks=[...], ...)
    def my_app(config):
        ...

"""
import logging
from abc import ABC, abstractmethod
from copy import copy
from unittest.mock import Mock

import matplotlib.animation
import matplotx.styles
import mlflow
import torch

import regelum
import pandas as pd

import time
import datetime
import dill
import os
import git
import shutil


import matplotlib.pyplot as plt

import pkg_resources

from pathlib import Path

import sys
import filelock
import regelum.__internal.base

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends import backend_qt5agg  # e.g.

from svgpathtools import svg2paths
from svgpath2mpl import parse_path


def is_in_debug_mode():
    return sys.gettrace() is not None


def passdown(CallbackClass):
    """Decorate a callback class in such a way that its event handling is inherited by derived classes.

    :param CallbackClass:
    :type CallbackClass: type
    :return: altered class that passes down its handlers to derived classes (regardless of whether handling methods are overriden)
    """

    class PassdownCallback(CallbackClass):
        def __call_passdown(self, obj, method, output):
            if (
                True
            ):  # self.ready(t):   # Currently, cooldowns don't work for PassdownCallbacks
                try:
                    if PassdownCallback.is_target_event(self, obj, method, output):
                        PassdownCallback.perform(self, obj, method, output)
                        self.on_trigger(PassdownCallback)
                        # self.trigger_cooldown(t)
                except regelum.RegelumExitException as e:
                    raise e
                except Exception as e:
                    self.log(
                        f"Callback {self.__class__.__name__} failed, when executing routines passed down from {CallbackClass.__name__}."
                    )
                    self.exception(e)

    PassdownCallback.__name__ = CallbackClass.__name__
    return PassdownCallback


class Callback(regelum.__internal.base.RegelumBase, ABC):
    """Base class for callbacks.

    Callback objects are used to perform in response to some method being called.
    """

    cooldown = None

    def __init__(self, log_level="info", attachee=None):
        """Initialize a callback object.

        :param logger: A logging object that will be used to log messages.
        :type logger: logging.Logger
        :param log_level: The level at which messages should be logged.
        :type log_level: str
        """
        super().__init__()
        self.attachee = attachee
        self.log = self._metadata["logger"].__getattribute__(log_level)
        # TODO: FIX THIS. Setting the level is needed due to the fact that mlflow sql backend reinstantiates logger
        # Moreover, rubbish mlflow backend logs are generated. They are not needed for a common user
        self._metadata["logger"].setLevel(logging.INFO)
        self.exception = self._metadata["logger"].exception
        self.__last_trigger = 0.0

    @classmethod
    def register(cls, *args, launch=False, **kwargs):
        existing_callbacks = [
            type(callback) for callback in cls._metadata["main"].callbacks
        ]
        if cls not in existing_callbacks:
            callback_instance = cls(*args, **kwargs)
            if launch:
                callback_instance.on_launch()
            cls._metadata["main"].callbacks = [callback_instance] + cls._metadata[
                "main"
            ].callbacks

    @abstractmethod
    def is_target_event(self, obj, method, output):
        pass

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        pass

    def on_trigger(self, caller):
        pass

    def on_iteration_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        pass

    def ready(self, t):
        if not self.cooldown:
            return True
        if t - self.__last_trigger > self.cooldown:
            return True
        else:
            return False

    def trigger_cooldown(self, t):
        self.__last_trigger = t

    def on_launch(self):
        pass

    @abstractmethod
    def perform(self, obj, method, output):
        pass

    def __call__(self, obj, method, output):
        t = time.time()
        if self.ready(t):
            try:
                if self.is_target_event(obj, method, output):
                    self.perform(obj, method, output)
                    self.on_trigger(self.__class__)
                    self.trigger_cooldown(t)
            except regelum.RegelumExitException as e:
                raise e
            except Exception as e:
                self.log(f"Callback {self.__class__.__name__} failed.")
                self.exception(e)
        for base in self.__class__.__bases__:
            if hasattr(base, f"_{base.__name__}__call_passdown"):
                base.__getattribute__(f"_{base.__name__}__call_passdown")(
                    self, obj, method, output
                )

    @classmethod
    def attach(cls, other):
        if hasattr(other, "_attached"):
            attached = other._attached + [cls]
        else:
            attached = [cls]

        class Attachee(other):
            _real_name = other.__name__
            _attached = attached

        return Attachee

    """
    @classmethod
    def detach(cls, other):
        if hasattr(other, "_attached"):
            attached = other._attached.copy()
            if cls not in attached:
                raise ValueError(f"Attempted to detach {cls.__name__}, but it was not attached.")
            attached.pop(attached.index(cls))
        else:
            raise ValueError(f"Attempted to detach {cls.__name__}, but it was not attached.")
        class Detachee(other):
            _real_name = other.__name__
            _attached = attached
        Detachee.__name__ = other.__name__
        return Detachee
    """

    def on_termination(self, res):
        pass


class TimeCallback(Callback):
    """Callback responsible for keeping track of simulation time."""

    def is_target_event(self, obj, method, output):
        return isinstance(obj, regelum.scenario.Scenario) and method == "post_step"

    def on_launch(self):
        regelum.main.metadata["time"] = 0.0

    def perform(self, obj, method, output):
        regelum.main.metadata["time"] = obj.time if not obj.is_episode_ended else 0.0


class OnEpisodeDoneCallback(Callback):
    """Callback responsible for logging and recording relevant data when an episode ends."""

    def __init__(self, *args, **kwargs):
        """Initialize an OnEpisodeDoneCallback instance."""
        super().__init__(*args, **kwargs)
        self.episode_counter = 0
        self.iteration_counter = 1

    def is_target_event(self, obj, method, output):
        return isinstance(obj, regelum.scenario.Scenario) and (
            method == "reload_pipeline"
        )

    def perform(self, obj, method, output):
        self.episode_counter += 1
        for callback in self._metadata["main"].callbacks:
            callback.on_episode_done(
                obj,
                self.episode_counter,
                obj.N_episodes,
                self.iteration_counter,
                obj.N_iterations,
            )

        if self.episode_counter == obj.N_episodes:
            self.episode_counter = 0
            self.iteration_counter += 1


class OnIterationDoneCallback(Callback):
    """Callback responsible for logging and recording relevant data when an iteration ends."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of OnIterationDoneCallback."""
        super().__init__(*args, **kwargs)
        self.iteration_counter = 0

    def is_target_event(self, obj, method, output):
        return isinstance(obj, regelum.scenario.Scenario) and (
            method == "reset_iteration"
        )

    def perform(self, obj, method, output):
        self.iteration_counter += 1
        if obj.N_iterations > 1 and self.iteration_counter <= obj.N_iterations:
            for callback in self._metadata["main"].callbacks:
                callback.on_iteration_done(
                    obj,
                    obj.N_episodes,
                    obj.N_episodes,
                    self.iteration_counter,
                    obj.N_iterations,
                )


class ConfigDiagramCallback(Callback):
    """Callback responsible for constructing SUMMARY.html and relevant data, including the config diagram."""

    def perform(self, *args, **kwargs):
        pass

    def is_target_event(self, obj, method, output):
        return False

    @staticmethod
    def __monitor_git():
        try:
            forbidden = False
            with filelock.FileLock(regelum.main.metadata["common_dir"] + "/diff.lock"):
                repo = git.Repo(
                    path=regelum.main.metadata["initial_working_directory"],
                    search_parent_directories=True,
                )
                commit_hash = repo.head.object.hexsha
                if repo.is_dirty(untracked_files=True):
                    commit_hash += (
                        ' <font color="red">(uncommitted/unstaged changes)</font>'
                    )
                    forbidden = (
                        "disallow_uncommitted" in regelum.main.config
                        and regelum.main.config.disallow_uncommitted
                        and not is_in_debug_mode()
                    )
                    untracked = repo.untracked_files
                    if not os.path.exists(
                        regelum.main.metadata["common_dir"] + "/changes.diff"
                    ):
                        repo.git.add(all=True)
                        with open(
                            regelum.main.metadata["common_dir"] + "/changes.diff", "w"
                        ) as f:
                            diff = repo.git.diff(repo.head.commit.tree)
                            if untracked:
                                repo.index.remove(untracked, cached=True)
                            f.write(diff + "\n")
                        with open(".summary/changes.diff", "w") as f:
                            f.write(diff + "\n")
                    else:
                        shutil.copy(
                            regelum.main.metadata["common_dir"] + "/changes.diff",
                            ".summary/changes.diff",
                        )
            if forbidden:
                raise Exception(
                    "Running experiments without committing is disallowed. Please, commit your changes."
                )
        except git.exc.InvalidGitRepositoryError as err:
            commit_hash = None
            repo = None
            if (
                "disallow_uncommitted" in regelum.main.cfg
                and regelum.main.cfg.disallow_uncommitted
                and not is_in_debug_mode()
            ):
                raise Exception(
                    "Running experiments without committing is disallowed. Please, commit your changes."
                ) from err
        return repo, commit_hash

    def on_launch(self):
        cfg = regelum.main.config
        metadata = regelum.main.metadata
        report = metadata["report"]
        start = time.time()
        os.mkdir(".summary")
        name = metadata["config_path"].split("/")[-1].split(".")[0]
        cfg.treemap(root=name).write_html("SUMMARY.html")
        with open("SUMMARY.html", "r") as f:
            html = f.read()
        repo, commit_hash = (
            self.__monitor_git()
            if not regelum.main.metadata["no_git"]
            else (None, None)
        )
        cfg_hash = hex(hash(cfg))
        html = html.replace("<body>", f"<title>{name} {cfg_hash}</title><body>")
        overrides_table = ""
        with open(".rehydra/rehydra.yaml", "r") as f:
            content = f.read()
            content = content[content.find("task:") :]
            content = content[: content.find("job:")]
            content = content.replace("-", "").split()[1:]
            if content[0] != "[]":
                with report() as r:
                    r["overrides"] = {}
                    for line in content:
                        field, value = line.split("=")
                        overrides_table += f'<tr><td><font face="Courier New">{field}</font></td> <td><font face="Courier New"> = </font></td>  <td><font face="Courier New">{value}</font></td> </tr>\n'
                        r["overrides"][field] = value
                    r["overrides_html"] = f"<table>{overrides_table}</table>"
            else:
                with report() as r:
                    r["overrides"] = {}
                    r["overrides_html"] = ""
        html = html.replace(
            "<head>",
            """
                            <head>
                              <link rel="mask-icon" type="image/x-icon" href="https://cpwebassets.codepen.io/assets/favicon/logo-pin-b4b4269c16397ad2f0f7a01bcdf513a1994f4c94b8af2f191c09eb0d601762b1.svg" color="#111" />  
                              <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/prism/1.22.0/themes/prism-tomorrow.min.css'>
                              
                            <style>
                            *,
                            *:before,
                            *:after {
                              box-sizing: border-box;
                            }
                            
                            pre[class*="language-"] {
                              position: relative;
                              overflow: auto;
                            
                              /* make space  */
                              margin: 5px 0;
                              padding: 1.75rem 0 1.75rem 1rem;
                              border-radius: 10px;
                            }
                            
                            pre[class*="language-"] button {
                              position: absolute;
                              top: 5px;
                              right: 5px;
                            
                              font-size: 0.9rem;
                              padding: 0.15rem;
                              background-color: #828282;
                            
                              border: ridge 1px #7b7b7c;
                              border-radius: 5px;
                              text-shadow: #c4c4c4 0 0 2px;
                            }
                            
                            pre[class*="language-"] button:hover {
                              cursor: pointer;
                              background-color: #bcbabb;
                            }
                            
                            main {
                              display: grid;
                              max-width: 600px;
                              margin: 20px auto;
                            }
                            
                            </style>
                            
                              <script>
                              window.console = window.console || function(t) {};
                            </script>
                              
                              <script>
                              if (document.location.search.match(/type=embed/gi)) {
                                window.parent.postMessage("resize", "*");
                              }
                            </script>
                            """,
        )
        html = html.replace(
            "<body>",
            f"""
                            <body>
                            <div>
                            <table style="margin-left: auto; margin-right: 0;">
                            <tbody>
                            {overrides_table}
                            </tbody>
                            </table>
                            </div>
                            </div>
            """,
        )
        html = html.replace(
            "<body>",
            f"""
                            <body>
                            <div style="display: grid; grid-template-columns: 1fr 1fr;">
                            <div>
                            <table>
                            <tbody>
                            <tr><td>Config hash:  </td> <td>{cfg_hash}</td></tr>
                            <tr><td>Commit hash: </td><td>{commit_hash} </td></tr>
                            <tr><td>Date and time: </td><td>{datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")} </td></tr>
                            <tr><td>Script path: </td><td>{metadata["script_path"]} </td></tr>
                            <tr><td>Config path: </td><td>{metadata["config_path"]}</td></tr>
                            </tbody>
                            </table>
                            </div>
            """,
        )
        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(
            [(i.key, i.version) for i in installed_packages]
        )
        packages_table = []
        for package, version in installed_packages_list:
            if package in [
                "matplotlib",
                "numpy",
                "scipy",
                "omegaconf",
                "hydra-core",
                "hydra-joblib-launcher",
                "pandas",
                "gpytorch",
                "torch",
                "casadi",
                "dill",
                "plotly",
                "regelum",
            ]:
                packages_table = [
                    f'<tr><td><font face="Courier New" color="red">{package}</font></td> <td><font face="Courier New" color="red"> == </font></td>  <td><font face="Courier New" color="red">{version}</font></td> </tr>\n'
                ] + packages_table
            else:
                packages_table.append(
                    f'<tr><td><font face="Courier New">{package}</font></td> <td><font face="Courier New"> == </font></td>  <td><font face="Courier New">{version}</font></td> </tr>\n'
                )

        html = html.replace(
            "</body>",
            f"""
                            <br>
                            <details>
                            <summary> Environment details </summary>
                            <table>
                            <tr style="vertical-align:top"> 
                            <td> <table>  {"".join(packages_table[:len(packages_table) // 4 ])}  </table> </td>
                            <td> <table>  {"".join(packages_table[len(packages_table) // 4 :2 * len(packages_table) // 4])}  </table> </td>
                            <td> <table>  {"".join(packages_table[2 *  len(packages_table) // 4: 3 * len(packages_table) // 4])}  </table> </td>
                            <td> <table>  {"".join(packages_table[3 *  len(packages_table) // 4:])}  </table> </td>
                            </tr>
                            </table>
                            </details>
                             </body>
            """,
        )
        git_fragment = (
            f"""git checkout {commit_hash.replace(' <font color="red">(uncommitted/unstaged changes)</font>',  chr(10) + f'patch -p1 < {os.path.abspath(".summary/changes.diff")}')}"""
            + chr(10)
            if commit_hash
            else ""
        )
        html = html.replace(
            "</body>",
            f"""
              <details>
              <summary>Snippets</summary>
              <main>
              <p>Extract callbacks:</p>
              <pre><code class="language-python">import dill, os, sys
os.chdir("{metadata["initial_working_directory"]}")
sys.path[:0] = {metadata["initial_pythonpath"].split(":")}
with open("{os.path.abspath(".")}/callbacks.dill", "rb") as f:
    callbacks = dill.load(f)</code></pre>"""
            + (
                f"""
              <p>Reproduce experiment:</p>
              <pre><code class="language-bash">cd {repo.working_tree_dir}
git reset
git restore .
git clean -f
{git_fragment}cd {metadata["initial_working_directory"]}
export PYTHONPATH="{metadata["initial_pythonpath"]}"
python3 {metadata["script_path"]} {" ".join(content if content[0] != "[]" else [])} {" ".join(list(filter(lambda x: "--" in x and "multirun" not in x, sys.argv)))} </code></pre>
            """
                if repo is not None
                else ""
            )
            + """</main>
                <script src="https://cpwebassets.codepen.io/assets/common/stopExecutionOnTimeout-2c7831bb44f98c1391d6a4ffda0e1fd302503391ca806e7fcc7b9b87197aec26.js"></script>
            
              <script src='https://cdnjs.cloudflare.com/ajax/libs/prism/1.26.0/prism.min.js'></script>
              <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.26.0/components/prism-python.min.js"></script>
              <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.26.0/components/prism-bash.min.js"></script>
                  <script>
            const copyButtonLabel = "Copy Code";
            
            // use a class selector if available
            let blocks = document.querySelectorAll("pre");
            
            blocks.forEach(block => {
              // only add button if browser supports Clipboard API
              if (navigator.clipboard) {
                let button = document.createElement("button");
            
                button.innerText = copyButtonLabel;
                block.appendChild(button);
            
                button.addEventListener("click", async () => {
                  await copyCode(block, button);
                });
              }
            });
            
            async function copyCode(block, button) {
              let code = block.querySelector("code");
              let text = code.innerText;
            
              await navigator.clipboard.writeText(text);
            
              // visual feedback that task is completed
              button.innerText = "Code Copied";
            
              setTimeout(() => {
                button.innerText = copyButtonLabel;
              }, 700);
            }
            //# sourceURL=pen.js
                </script>
            </details>
            </body>
            """,
        )
        with open("SUMMARY.html", "w") as f:
            f.write(html)
        self.log(
            f"Saved summary to {os.path.abspath('SUMMARY.html')}. ({int(1000 * (time.time() - start))}ms)"
        )

    def on_termination(self, res):
        with open("SUMMARY.html", "r") as f:
            html = f.read()
        table_lines = ""
        images = []
        directory = os.fsencode("gfx")

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            images.append(
                f'<img src="gfx/{filename}" style="object-fit: cover; width: 100%; max-height: 100%;">'
            )
        for image in images:
            table_lines += f"<div>{image}</div>\n"
        html = html.replace(
            "</body>",
            f"""
                            <br>
                            <div style="display: grid; grid-template-columns: 1fr 1fr;">
                            {table_lines}
                            </div>
                             </body>
                             """,
        )
        with open("SUMMARY.html", "w") as f:
            f.write(html)


plt.rcParams["animation.frame_format"] = "svg"  # VERY important

plt.style.use(matplotx.styles.dracula)


class AnimationCallback(Callback, ABC):
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

        :param animations: animation classes to be composed
        :param kwargs: keyword arguments to be passed to __init__ of said animations and the base class
        """
        Callback.__init__(self, **kwargs)
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
        try:
            super().lim(*args, extra_margin=extra_margin, **kwargs)
            return
        except ValueError:
            x, y = np.array([list(datum.values()) for datum in self.frame_data]).T
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_min, x_max = x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1
            y_min, y_max = y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1
            self.ax.set_xlim(
                x_min - extra_margin,
                x_min + max(x_max - x_min, y_max - y_min) + extra_margin,
            )
            self.ax.set_ylim(
                y_min - extra_margin,
                y_min + max(x_max - x_min, y_max - y_min) + extra_margin,
            )


@passdown
class StateTracker(Callback):
    """Records the state of the simulated system into `self.system_state`.

    Useful for animations that visualize motion of dynamical systems.
    """

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, regelum.simulator.Simulator)
            and method == "get_sim_step_data"
        )

    def perform(self, obj, method, output):
        self.system_state = obj.state


class PlanarMotionAnimation(PointAnimation, StateTracker):
    """Animates dynamics of systems that can be viewed as a point moving on a plane."""

    def on_trigger(self, _):
        self.add_frame(x=self.system_state[0], y=self.system_state[1])


class TriangleAnimation(AnimationCallback, ABC):
    """Animation that sets the location and rotation of a planar equilateral triangle at each frame."""

    _pic = None  # must be an svg located in regelum/img

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
        parsed.vertices -= parsed.vertices.mean(axis=0)
        self.marker = matplotlib.markers.MarkerStyle(marker=parsed)
        (self.triangle,) = self.ax.plot(0, 1, marker=self.marker, ms=30)

    def lim(self, *args, extra_margin=0.11, **kwargs):
        try:
            super().lim(*args, extra_margin=extra_margin, **kwargs)
            return
        except ValueError:
            x, y = np.array([list(datum.values()) for datum in self.frame_data]).T[:2]
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_min, x_max = x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1
            y_min, y_max = y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1
            self.ax.set_xlim(
                x_min - extra_margin,
                x_min + max(x_max - x_min, y_max - y_min) + extra_margin,
            )
            self.ax.set_ylim(
                y_min - extra_margin,
                y_min + max(x_max - x_min, y_max - y_min) + extra_margin,
            )

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
        self.marker._transform = self.marker.get_transform().rotate_deg(
            180 * theta / np.pi
        )
        return (self.triangle,)


class DirectionalPlanarMotionAnimation(TriangleAnimation, StateTracker):
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


class PendulumAnimation(PlanarMotionAnimation):
    """Animates the head of a swinging pendulum.

    Interprets the first state coordinate as the angle of the pendulum with respect to the topmost position.
    """

    def on_trigger(self, _):
        self.add_frame(x=np.sin(self.system_state[0]), y=np.cos(self.system_state[0]))

    def lim(self):
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)


class BarAnimation(AnimationCallback, StateTracker):
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


class HistoricalCallback(Callback, ABC):
    """Callback (base) responsible for recording various temporal data."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of HistoricalCallback."""
        super().__init__(*args, **kwargs)
        self.xlabel = "Time"
        self.ylabel = "Value"
        self.__data = pd.DataFrame()

        self.save_directory = Path(f".callbacks/{self.__class__.__name__}").resolve()

    def get_save_directory(self):
        return self.save_directory

    def on_launch(self):
        os.mkdir(self.get_save_directory())

    def add_datum(self, datum: dict):
        if self.__data.empty:
            self.__data = pd.DataFrame({key: [value] for key, value in datum.items()})
        else:
            self.__data.loc[len(self.__data)] = datum

    def clear_recent_data(self):
        self.__data = pd.DataFrame()

    def dump_data(self, identifier):
        if not self.__data.empty:
            self.__data.to_hdf(
                f".callbacks/{self.__class__.__name__}/{identifier}.h5", "data"
            )

    def load_data(self, idx=None):
        dirs = sorted(Path(self.get_save_directory()).iterdir())

        if len(dirs) > 0:
            if idx is None:
                return pd.concat(
                    [pd.read_hdf(path) for path in dirs],
                    axis=0,
                )
            else:
                if idx == "last":
                    idx = len(dirs)
                assert idx >= 1, "Indices should be no smaller than 1."
                assert idx <= len(dirs), f"Only {len(dirs)} files were stored."
                return pd.read_hdf(dirs[idx - 1])
        else:
            return pd.DataFrame()

    def insert_column_left(self, column_name, values):
        if not self.__data.empty:
            self.__data.insert(0, column_name, values)

    def dump_and_clear_data(self, identifier):
        self.dump_data(identifier)
        self.clear_recent_data()

    @property
    def data(self):
        return self.__data

    def plot(self, name=None):
        if regelum.main.is_clear_matplotlib_cache_in_callbacks:
            plt.clf()
            plt.cla()
            plt.close()
        if not name:
            name = self.__class__.__name__
        res = self.data.plot()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(name)
        plt.grid()
        # plt.xticks(range(1, len(self.data) + 1))
        return res.figure

    def save_plot(self, name=None):
        if not name:
            name = self.__class__.__name__
        self.plot(name=name)
        plt.savefig(f"gfx/{name}.svg")
        mlflow.log_artifact(f"gfx/{name}.svg", "gfx")

    def plot_gui(self):
        self.__data = self.load_data(idx="last")
        if not self.data.empty:
            return self.plot(name=self.__class__.__name__)
        else:
            return None


def method_callback(method_name, class_name=None, log_level="debug"):
    """Create a callback class that logs the output of a specific method of a class or any class.

    :param method_name: Name of the method to log output for.
    :type method_name: str
    :param class_name: (Optional) Name of the class the method belongs to. If not specified, the callback will log the output for the method of any class.
    :type class_name: str or class
    :param log_level: (Optional) The level of logging to use. Default is "debug".
    :type log_level: str

    """
    if class_name is not None:
        class_name = (
            class_name.__name__ if not isinstance(class_name, str) else class_name
        )

    class MethodCallback(Callback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def is_target_event(self, obj, method, output):
            return method == method_name and class_name in [
                None,
                obj.__class__.__name__,
            ]

        def perform(self, obj, method, output):
            self.log(
                f"Method '{method}' of class '{obj.__class__.__name__}' returned {output}"
            )

    return MethodCallback


class StateCallback(Callback):
    """StateCallback is a subclass of the Callback class that logs the state of a system object when the compute_closed_loop_rhs method of the System class is called.

    Attributes
    ----------
    log (function): A function that logs a message at a specified log level.
    """

    def is_target_event(self, obj, method, output):
        attachee = self.attachee if self.attachee is not None else regelum.system.System
        return (
            isinstance(obj, attachee)
            and method == regelum.system.System.compute_closed_loop_rhs.__name__
        )

    def perform(self, obj, method, output):
        self.log(f"System's state: {obj._state}")


class ObjectiveCallback(Callback):
    """A Callback class that logs the current objective value of an Actor instance.

    This callback is triggered whenever the Actor.objective method is called.

    Attributes
    ----------
    log (function): A logger function with the specified log level.
    """

    cooldown = 8.0

    def is_target_event(self, obj, method, output):
        return isinstance(obj, regelum.policies.Policy) and method == "objective"

    def perform(self, obj, method, output):
        self.log(f"Current objective: {output}")


class HistoricalObjectiveCallback(HistoricalCallback):
    """A callback which allows to store desired data collected among different runs inside multirun execution runtime."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of HistoricalObjectiveCallback."""
        super().__init__(*args, **kwargs)

        self.timeline = []
        self.num_launch = 1
        self.counter = 0
        self.cooldown = 1.0

    def on_launch(self):
        super().on_launch()
        with regelum.main.metadata["report"]() as r:
            r["elapsed_relative"] = 0

    def is_target_event(self, obj, method, output):
        return isinstance(obj, regelum.scenario.Scenario) and (method == "post_step")

    def perform(self, obj, method, output):
        self.counter += 1
        self.log(
            f"Current objective: {output[0]}, observation: {output[1][0]}, action: {output[2][0]}, total objective: {output[3]:.4f}, time: {obj.time:.4f} ({100 * obj.time/obj.simulator.time_final:.1f}%), episode: {obj.episode_counter + 1}/{obj.N_episodes}, iteration: {obj.iteration_counter + 1}/{obj.N_iterations}"
        )
        if not self.counter % 3:
            do_exit = False
            with regelum.main.metadata["report"]() as r:
                r["elapsed_relative"] = obj.time / obj.simulator.time_final
                if "terminate" in r:
                    do_exit = True
            if do_exit:
                self.log("Termination request issued from GUI.")
                raise regelum.RegelumExitException(
                    "Termination request issued from gui."
                )

        self.add_datum(
            {
                "time": round(output[3], 4),
                "current objective": output[0],
                "observation": output[1][0],
                "action": output[2][0],
                "total_objective": round(output[3], 4),
                "completed_percent": 100
                * round(obj.time / obj.simulator.time_final, 1),
            }
        )

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        self.insert_column_left("episode", episode_number)
        self.dump_and_clear_data(f"episode_{str(episode_number).zfill(5)}")


class SaveProgressCallback(Callback):
    """Callback responsible for regularly saving data collected by other callbacks to the hard-drive."""

    once_in = 1

    def on_launch(self):
        with regelum.main.metadata["report"]() as r:
            r["episode_current"] = 0

    def is_target_event(self, obj, method, output):
        return False

    def perform(self, obj, method, output):
        pass

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        start = time.time()
        if episode_number % self.once_in:
            return
        filename = "callbacks.dill"
        with open(filename, "wb") as f:
            dill.dump(regelum.main.callbacks, f)
        self.log(
            f"Saved callbacks to {os.path.abspath(filename)}. ({int(1000 * (time.time() - start))}ms)"
        )
        if scenario is not None:
            with regelum.main.metadata["report"]() as r:
                r["episode_current"] = episode_number
                if "episodes_total" not in r:
                    r["episode_total"] = episodes_total
                r["elapsed_relative"] = 1.0

    def on_termination(self, res):
        if isinstance(res, Exception):
            self.on_episode_done(None, 0, None, None, None)


class HistoricalObservationCallback(HistoricalCallback):
    """A callback which allows to store desired data collected among different runs inside multirun execution runtime."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of HistoricalObservationCallback."""
        super().__init__(*args, **kwargs)

        self.cooldown = 0.0

        self.current_episode = None
        self.dt_simulator_counter = 0
        self.time_threshold = 10

    def is_target_event(self, obj, method, output):
        if isinstance(obj, regelum.scenario.Scenario) and (method == "post_step"):
            self.dt_simulator_counter += 1
            return not self.dt_simulator_counter % self.time_threshold

    def perform(self, obj, method, output):
        self.add_datum(
            {
                **{
                    "time": obj.time,
                    "action": obj.action[0],
                },
                **dict(zip(obj.observation_components_naming, obj.observation[0])),
            }
        )

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        if iterations_total == 1:
            identifier = f"observations_ep_{str(episode_number).zfill(5)}"
        else:
            identifier = f"observations_it_{str(iteration_number).zfill(5)}_ep_{str(episode_number).zfill(5)}"
        self.save_plot(identifier)
        self.insert_column_left("episode", episode_number)
        self.dump_and_clear_data(identifier)

    def plot(self, name=None):
        if regelum.main.is_clear_matplotlib_cache_in_callbacks:
            plt.clf()
            plt.cla()
            plt.close()

        if not name:
            name = self.__class__.__name__
        res = (
            self.data.drop(["action"], axis=1)
            .set_index("time")
            .plot(subplots=True, grid=True, xlabel="time", title=name)
        )
        return res[0].figure


class ObjectiveLearningSaver(HistoricalCallback):
    """A callback which allows to store desired data collected among different runs inside multirun execution runtime."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of PolicyGradientObjectiveSaverCallback."""
        super().__init__(*args, **kwargs)

        self.cooldown = 0.0
        self.iteration_number = 1

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, regelum.controller.RLController)
            and (method == "pre_optimize")
        ) or (
            isinstance(obj, regelum.optimizable.Optimizable)
            and (method == "post_epoch")
        )

    def perform(self, obj, method, output):
        if isinstance(obj, regelum.controller.RLController) and (
            method == "pre_optimize"
        ):
            which, event, time, episode_counter, iteration_counter = output
            if which == "Critic":
                self.key = f"B. {which} objective. "
            else:
                self.key = f"C. {which} objective. "

            if event == "compute_action":
                if time is None:
                    raise ValueError("Time should be passed if one uses compute action")
                self.key += f"Time {str(round(time, 4))}. Ep {str(episode_counter).zfill(5)}. It {str(iteration_counter).zfill(5)}"
            elif event == "reset_episode":
                self.key += f"Ep {str(episode_counter).zfill(5)}. It {str(iteration_counter).zfill(5)}"
            elif event == "reset_iteration":
                self.key += f"It {str(iteration_counter).zfill(5)}"

        if isinstance(obj, regelum.optimizable.Optimizable) and (
            method == "post_epoch"
        ):
            epoch_idx, objective_values = output
            if len(objective_values) > 0:
                mlflow.log_metric(
                    self.key,
                    np.mean(objective_values),
                    step=epoch_idx,
                )

    def on_iteration_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        # self.dump_and_clear_data(
        #     f"critic_objective_on_iteration_{str(iteration_number).zfill(5)}"
        # )
        self.iteration_number = iteration_number + 1


class TotalObjectiveCallback(HistoricalCallback):
    """Callback that regularly logs total objective."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of TotalObjectiveCallback."""
        super().__init__(*args, **kwargs)
        self.cache = pd.DataFrame()
        self.xlabel = "Episode"
        self.ylabel = "Total objective"

    def is_target_event(self, obj, method, output):
        return False

    def on_iteration_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        mlflow.log_metric(
            "A. Avg total objectives",
            np.mean(scenario.recent_total_objectives_of_episodes),
            step=iteration_number,
        )
        mlflow.log_metric(
            "A. Med total objectives",
            np.median(scenario.recent_total_objectives_of_episodes),
            step=iteration_number,
        )
        mlflow.log_metric(
            "A. Std total objectives",
            np.std(scenario.recent_total_objectives_of_episodes),
            step=iteration_number,
        )

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        self.add_datum(
            {
                "episode": len(self.data) + 1,
                "objective": scenario.recent_total_objective,
            }
        )
        self.log(
            f"Final total objective of episode {self.data.iloc[-1]['episode']} is {round(self.data.iloc[-1]['objective'], 2)}"
        )
        self.dump_data(
            f"Total_Objectives_in_iteration_{str(iteration_number).zfill(5)}"
        )
        mlflow.log_metric(
            f"C. Total objectives in iteration {str(iteration_number).zfill(5)}",
            scenario.recent_total_objective,
            step=len(self.data),
        )

        self.save_plot(
            f"Total_Objectives_in_iteration_{str(iteration_number).zfill(5)}"
        )

        if episode_number == episodes_total:
            self.clear_recent_data()

    def perform(self, obj, method, output):
        pass

    def load_data(self, idx=None):
        return super().load_data(idx=1)

    def plot(self, name=None):
        if regelum.main.is_clear_matplotlib_cache_in_callbacks:
            plt.clf()
            plt.cla()
            plt.close()
        if not name:
            name = self.__class__.__name__

        res = self.data.set_index("episode").plot()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(name)
        plt.grid()
        plt.xticks(range(1, len(self.data) + 1))
        return res.figure


class TimeRemainingCallback(Callback):
    """Callback that logs an estimate of the time that remains till the end of the simulation."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of TimeRemainingCallback."""
        super().__init__(*args, **kwargs)
        self.time_episode = []

    def is_target_event(self, obj, method, output):
        return False

    def perform(self, obj, method, output):
        pass

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        self.time_episode.append(time.time())
        if len(self.time_episode) > 3:
            average_interval = 0
            previous = self.time_episode[-1]
            for current in self.time_episode[-2:-12:-1]:
                average_interval -= current - previous
                previous = current
            average_interval /= len(self.time_episode[-2:-12:-1])
            td = datetime.timedelta(
                seconds=int(
                    (iterations_total - iteration_number)
                    * average_interval
                    * (episodes_total - episode_number)
                )
            )
            remaining = f" Estimated remaining time: {str(td)}."
        else:
            remaining = ""
        self.log(
            f"Completed episode {episode_number}/{episodes_total} ({100*episode_number/episodes_total:.1f}%). Iteration: {iteration_number}/{iterations_total}."
            + remaining
        )


class CALFWeightsCallback(HistoricalCallback):
    """Callback that records the relevant model weights for CALF-based methods."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of CALFWeightsCallback."""
        super().__init__(*args, **kwargs)
        self.cooldown = 0.0
        self.time = 0.0

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, regelum.controller.Controller)
            and method == "compute_action"
            and "calf" in obj.critic.__class__.__name__.lower()
        )

    def perform(self, obj, method, output):
        try:
            datum = {
                **{"time": regelum.main.metadata["time"]},
                **{
                    f"weight_{i + 1}": weight[0]
                    for i, weight in enumerate(obj.critic.model.weights.full())
                },
            }

            # print(datum["time"], obj.critic.model.weights)
            self.add_datum(datum)
        finally:
            pass

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        identifier = f"CALF_weights_during_episode_{str(episode_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", episode_number)
            self.dump_and_clear_data(identifier)

    def plot(self, name=None):
        if not self.data.empty:
            if regelum.main.is_clear_matplotlib_cache_in_callbacks:
                plt.clf()
                plt.cla()
                plt.close()
            if not name:
                name = self.__class__.__name__
            res = self.data.set_index("time").plot(
                subplots=True, grid=True, xlabel="time", title=name
            )
            return res[0].figure


class CriticWeightsCallback(CALFWeightsCallback):
    """Whatever."""

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, regelum.controller.Controller)
            and method == "compute_action"
        )

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        identifier = f"Critic_weights_during_episode_{str(episode_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", episode_number)
            self.dump_and_clear_data(identifier)


class CalfCallback(HistoricalCallback):
    """Callback that records various diagnostic data during experiments with CALF-based methods."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of CalfCallback."""
        super().__init__(*args, **kwargs)
        self.cooldown = 1.0
        self.time = 0.0

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, regelum.controller.Controller)
            and method == "compute_action"
            and "calf" in obj.critic.__class__.__name__.lower()
        )

    def perform(self, obj, method, output):
        current_CALF = obj.critic(
            obj.critic.observation_last_good,
            use_stored_weights=True,
        )
        self.log(
            f"current CALF value:{current_CALF}, decay_rate:{obj.critic.safe_decay_rate}, observation: {obj.data_buffer.sample_last(keys=['observation'], dtype=np.array)['observation']}"
        )
        is_calf = (
            obj.critic.opt_status == "success" and obj.policy.opt_status == "success"
        )
        if not self.data.empty:
            prev_CALF = self.data["J_hat"].iloc[-1]
        else:
            prev_CALF = current_CALF

        delta_CALF = prev_CALF - current_CALF
        self.add_datum(
            {
                "time": regelum.main.metadata["time"],
                "J_hat": current_CALF[0]
                if isinstance(current_CALF, (np.ndarray, torch.Tensor))
                else current_CALF.full()[0],
                "is_CALF": is_calf,
                "delta": delta_CALF[0]
                if isinstance(delta_CALF, (np.ndarray, torch.Tensor))
                else delta_CALF.full()[0],
            }
        )

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        identifier = f"CALF_diagnostics_on_episode_{str(episode_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", episode_number)
            self.dump_and_clear_data(identifier)

    def plot(self, name=None):
        if regelum.main.is_clear_matplotlib_cache_in_callbacks:
            plt.clf()
            plt.cla()
            plt.close()
        if not self.data.empty:
            fig = plt.figure(figsize=(10, 10))

            ax_calf, ax_switch, ax_delta = ax_array = fig.subplots(1, 3)
            ax_calf.plot(self.data["time"], self.data["J_hat"], label="CALF")
            ax_switch.plot(self.data["time"], self.data["is_CALF"], label="CALF on")
            ax_delta.plot(self.data["time"], self.data["delta"], label="delta decay")

            for ax in ax_array:
                ax.set_xlabel("Time [s]")
                ax.grid()
                ax.legend()

            ax_calf.set_ylabel("J_hat")
            ax_switch.set_ylabel("Is CALF on")
            ax_delta.set_ylabel("Decay size")

            plt.legend()
            return fig


class CriticCallback(CalfCallback):
    """Watever."""

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, regelum.controller.Controller)
            and method == "compute_action"
        )

    def perform(self, obj, method, output):
        last_observation = obj.data_buffer.sample_last(
            keys=["observation"], dtype=np.array
        )["observation"]
        critic_val = obj.critic(
            last_observation,
        )
        self.log(f"current critic value:{critic_val}, observation: {last_observation}")
        if critic_val is not None:
            self.add_datum(
                {
                    "time": regelum.main.metadata["time"],
                    "J": critic_val[0]
                    if isinstance(critic_val, (np.ndarray, torch.Tensor))
                    else critic_val.full()[0],
                }
            )

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        identifier = f"Critic_values_on_episode_{str(episode_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", episode_number)
            self.dump_and_clear_data(identifier)

    def plot(self, name=None):
        if regelum.main.is_clear_matplotlib_cache_in_callbacks:
            plt.clf()
            plt.cla()
            plt.close()
        if not self.data.empty:
            fig = plt.figure(figsize=(10, 10))

            ax_critic = fig.subplots(1, 1)
            ax_critic.plot(self.data["time"], self.data["J"], label="Critic")

            ax_critic.set_xlabel("Time [s]")
            ax_critic.grid()
            ax_critic.legend()

            ax_critic.set_ylabel("J")

            plt.legend()
            return fig
