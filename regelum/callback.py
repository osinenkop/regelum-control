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

import mlflow

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

import numpy as np
from typing import Dict, Any

from regelum.event import Event
import torch


def is_in_debug_mode():
    return sys.gettrace() is not None


def disable_in_jupyter(method):
    def wrapper(self: Callback, *args, **kwargs):
        if self.is_jupyter:
            return lambda *args, **kwargs: None
        else:
            return method(self, *args, **kwargs)

    return wrapper


def detach(Attachee):
    """Create a duplicate of the provided regelum type with all callbacks detached from it."""

    if not hasattr(Attachee, "_attached"):
        return Attachee

    class Detachee(Attachee):
        _attached = []
        _real_name = Attachee.__name__

    return Detachee


def trigger(CallbackClass):
    """Decorate a callback class in such a way that its event handling is inherited by derived classes.

    Args:
        CallbackClass (type): class to be decorated

    Returns:
        CallbackClass: altered class that passes down its handlers to derived classes
            (regardless of whether handling methods are overriden)
    """

    class PassdownCallback(CallbackClass):
        def __call_passdown(self, obj, method, output):
            if (
                True
            ):  # self.ready(t):   # Currently, cooldowns don't work for PassdownCallbacks
                try:
                    if CallbackClass.is_target_event(self, obj, method, output, []):
                        CallbackClass.on_function_call(self, obj, method, output)
                        return True
                    return False
                    # self.on_trigger(PassdownCallback)
                    # self.trigger_cooldown(t)
                except regelum.RegelumExitException as e:
                    raise e
                except Exception as e:
                    self.log(
                        f"Callback {self.__class__.__name__} failed, when executing routines passed down from {CallbackClass.__name__}."
                    )
                    self.exception(e)
                    return False

        def on_function_call(self, *args, **kwargs):
            pass

    PassdownCallback.__name__ = CallbackClass.__name__
    return PassdownCallback


def is_interactive():
    import __main__ as main
    import sys

    return (
        (not hasattr(main, "__file__"))
        and (not is_in_debug_mode())
        and (not ("pytest" in sys.modules))
    )


class Callback(regelum.__internal.base.RegelumBase, ABC):
    """Base class for callbacks.

    Callback objects are used to perform in response to some method being called.
    """

    cooldown = None
    is_jupyter = is_interactive()

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
    def is_target_event(self, obj, method, output, triggers):
        pass

    def is_done_collecting(self):
        return True

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
    def on_function_call(self, obj, method, output):
        pass

    def __call__(self, obj, method, output):
        t = time.time()
        triggers = []
        for base in self.__class__.__bases__:
            if hasattr(base, f"_PassdownCallback__call_passdown"):
                if base._PassdownCallback__call_passdown(self, obj, method, output):
                    triggers.append(base)
        if self.ready(t):
            try:
                if self.is_target_event(obj, method, output, triggers):
                    self.on_function_call(obj, method, output)
                    done = True
                    for base in self.__class__.__bases__:
                        if hasattr(base, f"_PassdownCallback__call_passdown"):
                            done = done and base.is_done_collecting(self)
                    done = done and self.is_done_collecting()
                    if done and triggers:
                        self.on_trigger(self.__class__)
                    self.trigger_cooldown(t)
            except regelum.RegelumExitException as e:
                raise e
            except Exception as e:
                self.log(f"Callback {self.__class__.__name__} failed.")
                self.exception(e)

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


class OnEpisodeDoneCallback(Callback):
    """Callback responsible for logging and recording relevant data when an episode ends."""

    def __init__(self, *args, **kwargs):
        """Initialize an OnEpisodeDoneCallback instance."""
        super().__init__(*args, **kwargs)
        self.episode_counter = 0
        self.iteration_counter = 1

    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, regelum.scenario.Scenario) and (
            method == "reload_scenario"
        )

    def on_function_call(self, obj, method, output):
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

    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, regelum.scenario.Scenario) and (
            method == "reset_iteration"
        )

    def on_function_call(self, obj, method, output):
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

    def on_function_call(self, *args, **kwargs):
        pass

    def is_target_event(self, obj, method, output, triggers):
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
                "disallow_uncommitted" in regelum.main.config
                and regelum.main.config.disallow_uncommitted
                and not is_in_debug_mode()
            ):
                raise Exception(
                    "Running experiments without committing is disallowed. Please, commit your changes."
                ) from err
        return repo, commit_hash

    @disable_in_jupyter
    def on_launch(self):
        cfg = regelum.main.config
        metadata = regelum.main.metadata
        report = metadata["report"]
        start = time.time()
        os.mkdir(".summary")
        name = metadata["config_path"].split("/")[-1].split(".")[0] if metadata["config_path"] is not None else "None"
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


@trigger
class StateTracker(Callback):
    """Records the state of the simulated system into `self.system_state`.

    Useful for animations that visualize motion of dynamical systems.
    """

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.Scenario)
            and method == "post_compute_action"
        )

    def is_done_collecting(self):
        return hasattr(self, "system_state")

    def on_function_call(self, obj, method, output):
        self.system_state = obj.state
        self.system_state = self.system_state.reshape(self.system_state.size)
        self.state_naming = obj.simulator.system.state_naming


@trigger
class TimeTracker(Callback):
    """Records the state of the simulated system into `self.system_state`.

    Useful for animations that visualize motion of dynamical systems.
    """

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.Scenario)
            and method == "post_compute_action"
        )

    def is_done_collecting(self):
        return hasattr(self, "time")

    def on_function_call(self, obj, method, output):
        self.time = output["time"]


@trigger
class ObservationTracker(Callback):
    """Records the state of the simulated system into `self.system_state`.

    Useful for animations that visualize motion of dynamical systems.
    """

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.Scenario)
            and method == "post_compute_action"
        )

    def is_done_collecting(self):
        return hasattr(self, "observation")

    def on_function_call(self, obj, method, output):
        self.observation = output["observation"][0]
        self.observation_naming = obj.simulator.system.observation_naming


@trigger
class ObjectiveTracker(Callback):
    """Records the state of the simulated system into `self.system_state`.

    Useful for animations that visualize motion of dynamical systems.
    """

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.Scenario)
            and method == "post_compute_action"
        )

    def is_done_collecting(self):
        return hasattr(self, "objective")

    def on_function_call(self, obj, method, output):
        self.running_objective = output["running_objective"]
        self.value = output["current_value"]
        self.objective = np.array([self.value, self.running_objective])
        self.objective_naming = ["Value", "Running objective"]


@trigger
class ValueTracker(Callback):
    """Records the state of the simulated system into `self.system_state`.

    Useful for animations that visualize motion of dynamical systems.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = None

    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, regelum.scenario.Scenario) and (
            method == "reload_scenario"
        )

    def is_done_collecting(self):
        return hasattr(self, "score")

    def on_function_call(self, obj, method, output):
        self.score = obj.recent_value


@trigger
class ActionTracker(Callback):
    """Records the state of the simulated system into `self.system_state`.

    Useful for animations that visualize motion of dynamical systems.
    """

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.Scenario)
            and method == "post_compute_action"
        )

    def is_done_collecting(self):
        return hasattr(self, "action")

    def on_function_call(self, obj, method, output):
        self.action = output["action"][0]
        self.action_naming = obj.simulator.system._inputs_naming


class HistoricalCallback(Callback, ABC):
    """Callback (base) responsible for recording various temporal data."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of HistoricalCallback."""
        super().__init__(*args, **kwargs)
        self.xlabel = "Time"
        self.ylabel = "Value"
        self.data = pd.DataFrame()

        self.save_directory = Path(f".callbacks/{self.__class__.__name__}").resolve()

    def get_save_directory(self):
        return self.save_directory

    def on_launch(self):
        os.mkdir(self.get_save_directory())

    def add_datum(self, datum: dict):
        if self.data.empty:
            self.data = pd.DataFrame({key: [value] for key, value in datum.items()})
        else:
            self.data.loc[len(self.data)] = datum

    @disable_in_jupyter
    def dump_data(self, identifier):
        if not self.data.empty:
            self.data.to_hdf(
                f".callbacks/{self.__class__.__name__}/{identifier}.h5", key="data"
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
        if not self.data.empty:
            self.data.insert(0, column_name, values)

    @disable_in_jupyter
    def dump_and_clear_data(self, identifier):
        self.dump_data(identifier)
        self.clear_recent_data()

    def clear_recent_data(self):
        self.data = pd.DataFrame()

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
        self.data = self.load_data(idx="last")
        if not self.data.empty:
            return self.plot(name=self.__class__.__name__)
        else:
            return None


class CalfCallback(HistoricalCallback):
    """Callback that records various diagnostic data during experiments with CALF-based methods."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of CalfCallback."""
        super().__init__(*args, **kwargs)
        self.cooldown = 1.0
        self.time = 0.0

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, regelum.scenario.Scenario)
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
            obj.critic.opt_status == regelum.optimizable.optimizers.OptStatus.success
            and obj.policy.opt_status
            == regelum.optimizable.optimizers.OptStatus.success
        )
        if not self.data.empty:
            prev_CALF = self.data["J_hat"].iloc[-1]
        else:
            prev_CALF = current_CALF

        delta_CALF = prev_CALF - current_CALF
        self.add_datum(
            {
                "time": obj.data_buffer.get_latest("time"),
                "J_hat": (
                    current_CALF[0]
                    if isinstance(current_CALF, (np.ndarray, torch.Tensor))
                    else current_CALF.full()[0]
                ),
                "is_CALF": is_calf,
                "delta": (
                    delta_CALF[0]
                    if isinstance(delta_CALF, (np.ndarray, torch.Tensor))
                    else delta_CALF.full()[0]
                ),
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
        identifier = f"CALF_diagnostics_on_episode_{str(iteration_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("iteration", iteration_number)
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


class CALFWeightsCallback(HistoricalCallback):
    """Callback that records the relevant model weights for CALF-based methods."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of CALFWeightsCallback."""
        super().__init__(*args, **kwargs)
        self.cooldown = 0.0
        self.time = 0.0

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, regelum.scenario.Scenario)
            and method == "compute_action"
            and "calf" in obj.critic.__class__.__name__.lower()
        )

    def perform(self, obj, method, output):
        try:
            datum = {
                **{
                    "time": obj.data_buffer.get_latest("time"),
                },
                **{
                    f"weight_{i + 1}": weight
                    for i, weight in enumerate(
                        obj.critic.model.weights
                        if isinstance(
                            obj.critic.model.weights, (np.ndarray, torch.Tensor)
                        )
                        else obj.critic.model.weights.full().reshape(-1)
                    )
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
        identifier = f"CALF_weights_during_episode_{str(iteration_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", iteration_number)
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


def method_callback(method_name, class_name=None, log_level="debug"):
    """Create a callback class that logs the output of a specific method of a class or any class.

    Args:
        method_name (str): Name of the method to log output for.
        class_name (str or class): (Optional) Name of the class the
            method belongs to. If not specified, the callback will log
            the output for the method of any class.
        log_level (str): (Optional) The level of logging to use. Default
            is "debug".
    """
    if class_name is not None:
        class_name = (
            class_name.__name__ if not isinstance(class_name, str) else class_name
        )

    class MethodCallback(Callback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def is_target_event(self, obj, method, output, triggers):
            return method == method_name and class_name in [
                None,
                obj.__class__.__name__,
            ]

        def on_function_call(self, obj, method, output):
            self.log(
                f"Method '{method}' of class '{obj.__class__.__name__}' returned {output}"
            )

    return MethodCallback


class ScenarioStepLogger(Callback):
    """A callback which allows to log every step of simulation in a scenario."""

    cooldown = 1.0

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.Scenario)
            and method == "post_compute_action"
        )

    def on_function_call(self, obj, method: str, output: Dict[str, Any]):
        with np.printoptions(precision=2, suppress=True):
            self.log(
                f"runn. objective: {output['running_objective']:.2f}, "
                f"state est.: {output['estimated_state'][0]}, "
                f"observation: {output['observation'][0]}, "
                f"action: {output['action'][0]}, "
                f"value: {output['current_value']:.4f}, "
                f"time: {output['time']:.4f} ({100 * output['time']/obj.simulator.time_final:.1f}%), "
                f"episode: {int(output['episode_id'])}/{obj.N_episodes}, "
                f"iteration: {int(output['iteration_id'])}/{obj.N_iterations}"
            )


class SaveProgressCallback(Callback):
    """Callback responsible for regularly saving data collected by other callbacks to the hard-drive."""

    once_in = 1

    def on_launch(self):
        with regelum.main.metadata["report"]() as r:
            r["episode_current"] = 0

    def is_target_event(self, obj, method, output, triggers):
        return False

    def on_function_call(self, obj, method, output):
        pass

    @disable_in_jupyter
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


class HistoricalDataCallback(HistoricalCallback):
    """A callback which allows to store desired data collected among different runs inside multirun execution runtime."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of HistoricalDataCallback."""
        super().__init__(*args, **kwargs)
        self.cooldown = 0.0

        self.observation_components_naming = None
        self.action_components_naming = None
        self.state_components_naming = None

    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, regelum.scenario.Scenario) and (
            method == "post_compute_action" or method == "dump_data_buffer"
        )

    def on_function_call(self, obj, method, output):
        if self.observation_components_naming is None:
            self.observation_components_naming = (
                [
                    f"observation_{i + 1}"
                    for i in range(obj.simulator.system.dim_observation)
                ]
                if obj.simulator.system.observation_naming is None
                else obj.simulator.system.observation_naming
            )

        if self.action_components_naming is None:
            self.action_components_naming = (
                [f"action_{i + 1}" for i in range(obj.simulator.system.dim_inputs)]
                if obj.simulator.system.inputs_naming is None
                else obj.simulator.system.inputs_naming
            )

        if self.state_components_naming is None:
            self.state_components_naming = (
                [f"state_{i + 1}" for i in range(obj.simulator.system.dim_state)]
                if obj.simulator.system.state_naming is None
                else obj.simulator.system.state_naming
            )

        if method == "post_compute_action":
            self.add_datum(
                {
                    **{
                        "time": output["time"],
                        "running_objective": output["running_objective"],
                        "current_value": output["current_value"],
                        "episode_id": output["episode_id"],
                        "iteration_id": output["iteration_id"],
                    },
                    **dict(zip(self.action_components_naming, output["action"][0])),
                    **dict(
                        zip(self.state_components_naming, output["estimated_state"][0])
                    ),
                    # **dict(
                    #     zip(self.state_components_naming, output["estimated_state"][0])
                    # ),
                }
            )
        elif method == "dump_data_buffer":
            _, data_buffer = output
            self.data = pd.concat(
                [
                    data_buffer.to_pandas(
                        keys={
                            "time": float,
                            "running_objective": float,
                            "current_value": float,
                            "episode_id": int,
                            "iteration_id": int,
                        }
                    )
                ]
                + [
                    pd.DataFrame(
                        columns=columns,
                        data=np.array(
                            data_buffer.to_pandas([key]).values.tolist(),
                            dtype=float,
                        ).squeeze(),
                    )
                    for columns, key in [
                        (self.action_components_naming, "action"),
                        (self.state_components_naming, "estimated_state"),
                        # (self.state_components_naming, "estimated_state"),
                    ]
                ],
                axis=1,
            )

    @disable_in_jupyter
    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        if episodes_total == 1:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}"
        else:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}_ep_{str(episode_number).zfill(5)}"
        self.save_plot(identifier)
        self.dump_and_clear_data(identifier)

    def plot(self, name=None):
        if regelum.main.is_clear_matplotlib_cache_in_callbacks:
            plt.clf()
            plt.cla()
            plt.close()

        if not name:
            name = self.__class__.__name__

        axes = (
            self.data[
                self.state_components_naming + self.action_components_naming + ["time"]
            ]
            .set_index("time")
            .plot(subplots=True, grid=True, xlabel="time", title=name, legend=False)
        )
        for ax, label in zip(
            axes, self.state_components_naming + self.action_components_naming
        ):
            ax.set_ylabel(label)

        return axes[0].figure


class ObjectiveSaver(HistoricalCallback):
    """A callback which allows to store objective values during execution runtime."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of PolicyGradientObjectiveSaverCallback."""
        super().__init__(*args, **kwargs)

        self.cooldown = 0.0
        self.iteration_number = 1

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.RLScenario) and (method == "pre_optimize")
        ) or (
            isinstance(obj, regelum.optimizable.Optimizable)
            and (method == "post_epoch" or method == "post_optimize")
        )

    def on_function_call(self, obj, method, output):
        if isinstance(obj, regelum.scenario.RLScenario) and (method == "pre_optimize"):
            which, event, time, episode_counter, iteration_counter = output
            time = float(time)
            if which == "Critic":
                self.key = f"B. {which} objective. "
            else:
                self.key = f"C. {which} objective. "

            if event == Event.compute_action:
                if time is None:
                    raise ValueError("Time should be passed if one uses compute action")
                self.key += f"Time {str(round(time, 4))}. Ep {str(episode_counter).zfill(5)}. It {str(iteration_counter).zfill(5)}"
            elif event == Event.reset_episode:
                self.key += f"Ep {str(episode_counter).zfill(5)}. It {str(iteration_counter).zfill(5)}"
            elif event == Event.reset_iteration:
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
                self.add_datum(
                    {"epoch_idx": epoch_idx, "objective": np.mean(objective_values)}
                )

        if isinstance(obj, regelum.optimizable.Optimizable) and (
            method == "post_optimize"
        ):
            self.dump_and_clear_data(self.key)


class CriticObjectiveSaver(ObjectiveSaver):
    """A callback which allows to store critic objective values during execution runtime."""

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.RLScenario) and (method == "pre_optimize")
        ) or (
            isinstance(obj, regelum.critic.Critic)
            and (method == "post_epoch" or method == "post_optimize")
        )


class PolicyObjectiveSaver(ObjectiveSaver):
    """A callback which allows to store critic objective values during execution runtime."""

    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, regelum.scenario.RLScenario) and (method == "pre_optimize")
        ) or (
            isinstance(obj, regelum.policy.Policy)
            and (method == "post_epoch" or method == "post_optimize")
        )


class ValueCallback(HistoricalCallback):
    """Callback that regularly logs value."""

    def __init__(self, *args, **kwargs):
        """Initialize an instance of TotalObjectiveCallback."""
        super().__init__(*args, **kwargs)
        self.cache = pd.DataFrame()
        self.xlabel = "Episode"
        self.ylabel = "value"
        self.values = []
        self.mean_iteration_values = []

    def is_target_event(self, obj, method, output, triggers):
        return False

    @disable_in_jupyter
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
                "objective": scenario.recent_value,
            }
        )
        self.values.append(scenario.recent_value)
        self.log(
            f"Final value of episode {self.data.iloc[-1]['episode']} is {round(self.data.iloc[-1]['objective'], 2)}"
        )
        if not self.is_jupyter:
            self.dump_data(
                f"Total_Objectives_in_iteration_{str(iteration_number).zfill(5)}"
            )
            mlflow.log_metric(
                f"C. values in iteration {str(iteration_number).zfill(5)}",
                scenario.recent_value,
                step=len(self.values),
            )

            self.save_plot(
                f"Total_Objectives_in_iteration_{str(iteration_number).zfill(5)}"
            )

        if episode_number == episodes_total:
            if not self.is_jupyter:
                mlflow.log_metric(
                    "A. Avg value",
                    np.mean(self.values),
                    step=iteration_number,
                )
                self.mean_iteration_values.append(np.mean(self.values))
                self.values = []

                mlflow.log_metric(
                    "A. Cum max value",
                    np.max(self.mean_iteration_values),
                    step=iteration_number,
                )
                mlflow.log_metric(
                    "A. Cum min value",
                    np.min(self.mean_iteration_values),
                    step=iteration_number,
                )
            else:
                self.log(f"Avg value: {np.mean(self.values)}")

            self.clear_recent_data()

    def on_function_call(self, obj, method, output):
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

    def is_target_event(self, obj, method, output, triggers):
        return False

    def on_function_call(self, obj, method, output):
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
            episodes_left = (
                episodes_total
                - episode_number
                + (iterations_total - iteration_number) * episodes_total
            )
            td = datetime.timedelta(seconds=int(episodes_left * average_interval))
            remaining = f" Estimated remaining time: {str(td)}."
        else:
            remaining = ""
        self.log(
            f"Completed episode {episode_number}/{episodes_total} ({100*episode_number/episodes_total:.1f}%). Iteration: {iteration_number}/{iterations_total}."
            + remaining
        )


from .animation import *
