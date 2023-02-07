"""
This module contains callbacks.
Callbacks are lightweight event handlers, used mainly for logging.

To make a method of a class trigger callbacks, one needs to
decorate the class (or its base class) with ``@introduce_callbacks()`` and then
also decorate the method with ``@apply_callbacks``.

Callbacks can be registered by simply supplying them in the respective keyword argument:
::

    @rcognita.main(callbacks=[...], ...)
    def my_app(config):
        ...

"""

from abc import ABC, abstractmethod
import torch
import rcognita
import pandas as pd
import time, datetime
import dill, os, git, shutil

import omegaconf, json

import matplotlib.pyplot as plt
import re

import pkg_resources

from pathlib import Path

import sys
import filelock


def is_in_debug_mode():
    return not sys.gettrace() is None


def apply_callbacks(method):
    """
    Decorator that applies a list of callbacks to a given method of an object.
    If the object has no list of callbacks specified, the default callbacks are used.

    :param method: The method to which the callbacks should be applied.
    """

    def new_method(self, *args, **kwargs):
        res = method(self, *args, **kwargs)
        if self.callbacks is None:
            self.callbacks = rcognita.main.callbacks
        for callback in self.callbacks:
            callback(obj=self, method=method.__name__, output=res)
        return res

    return new_method


class introduce_callbacks:
    """
    A class decorator that introduces a `callbacks` attribute to the decorated class.
    The `callbacks` attribute is a list of callbacks that can be applied to methods
    of instances of the decorated class (for instance via `@apply_callbacks`).
    """

    def __init__(self, default_callbacks=None):
        """
        Initializes the decorator.

        :param default_callbacks: A list of callbacks that will be used as the default
            value for the `callbacks` attribute of instances of the decorated class.
            If no value is specified, the `callbacks` attribute will be initialized to `None`, which will
            in turn make `@apply_callbacks` use default callbacks instead.
        """
        self.default_callbacks = default_callbacks

    def __call__(self, cls):
        class whatever(cls):
            def __init__(self2, *args, callbacks=self.default_callbacks, **kwargs):
                super().__init__(*args, **kwargs)
                self2.callbacks = callbacks

        return whatever


class Callback(ABC):
    cooldown = None
    """
    This is the base class for a callback object. Callback objects are used to perform some action or computation after a method has been called.
    """

    def __init__(self, logger, log_level="info"):
        """
        Initialize a callback object.

        :param logger: A logging object that will be used to log messages.
        :type logger: logging.Logger
        :param log_level: The level at which messages should be logged.
        :type log_level: str
        """
        self.log = logger.__getattribute__(log_level)
        self.exception = logger.exception
        self.last_trigger = 0.0

    @abstractmethod
    def is_target_event(self, obj, method, output):
        pass

    def on_episode_done(self, scenario, episode_number, episodes_total):
        pass

    def ready(self, t):
        if not self.cooldown:
            return True
        if t - self.last_trigger > self.cooldown:
            return True
        else:
            return False

    def trigger_cooldown(self, t):
        self.last_trigger = t

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
                    self.trigger_cooldown(t)
            except rcognita.RcognitaExitException as e:
                raise e
            except Exception as e:
                self.log(f"Callback {self.__class__.__name__} failed.")
                self.exception(e)

    def on_termination(self, res):
        pass


class TimeCallback(Callback):
    def is_target_event(self, obj, method, output):
        return isinstance(obj, rcognita.scenarios.Scenario) and method == "post_step"

    def on_launch(self):
        rcognita.main.metadata["time"] = 0.0

    def perform(self, obj, method, output):
        rcognita.main.metadata["time"] = obj.time


class EventCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_counter = 0

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, rcognita.scenarios.Scenario) and method == "reload_pipeline"
        )

    def perform(self, obj, method, output):
        self.episode_counter += 1
        for callback in rcognita.main.callbacks:
            callback.on_episode_done(obj, self.episode_counter, obj.N_episodes)


class ConfigDiagramCallback(Callback):
    def perform(self, *args, **kwargs):
        pass

    def is_target_event(self, obj, method, output):
        return False

    def on_launch(self):
        cfg = rcognita.main.config
        metadata = rcognita.main.metadata
        report = metadata["report"]
        start = time.time()
        os.mkdir(".summary")
        name = metadata["config_path"].split("/")[-1].split(".")[0]
        cfg.treemap(root=name).write_html("SUMMARY.html")
        with open("SUMMARY.html", "r") as f:
            html = f.read()
        try:
            forbidden = False
            with filelock.FileLock(metadata["common_dir"] + "/diff.lock"):
                repo = git.Repo(search_parent_directories=True)
                commit_hash = repo.head.object.hexsha
                if repo.is_dirty(untracked_files=True):
                    commit_hash += (
                        ' <font color="red">(uncommitted/unstaged changes)</font>'
                    )
                    forbidden = (
                        "disallow_uncommitted" in cfg
                        and cfg.disallow_uncommitted
                        and not is_in_debug_mode()
                    )
                    untracked = repo.untracked_files
                    if not os.path.exists(metadata["common_dir"] + "/changes.diff"):
                        repo.git.add(all=True)
                        with open(metadata["common_dir"] + "/changes.diff", "w") as f:
                            diff = repo.git.diff(repo.head.commit.tree)
                            if untracked:
                                repo.index.remove(untracked, cached=True)
                            f.write(diff + "\n")
                        with open(".summary/changes.diff", "w") as f:
                            f.write(diff + "\n")
                    else:
                        shutil.copy(
                            metadata["common_dir"] + "/changes.diff",
                            ".summary/changes.diff",
                        )
            if forbidden:
                raise Exception(
                    "Running experiments without committing is disallowed. Please, commit your changes."
                )
        except git.exc.InvalidGitRepositoryError:
            commit_hash = None
            if (
                "disallow_uncommitted" in cfg
                and cfg.disallow_uncommitted
                and not is_in_debug_mode()
            ):
                raise Exception(
                    "Running experiments without committing is disallowed. Please, commit your changes."
                )
        cfg_hash = hex(hash(cfg))
        html = html.replace("<body>", f"<title>{name} {cfg_hash}</title><body>")
        overrides_table = ""
        with open(".hydra/hydra.yaml", "r") as f:
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
                "rcognita",
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
    callbacks = dill.load(f)</code></pre>
              <p>Reproduce experiment:</p>
              <pre><code class="language-bash">cd {repo.working_tree_dir}
git reset
git restore .
git clean -f
{f'''git checkout {commit_hash.replace(' <font color="red">(uncommitted/unstaged changes)</font>',  chr(10) + f'patch -p1 < {os.path.abspath(".summary/changes.diff")}')}''' + chr(10) if commit_hash else ""}cd {metadata["initial_working_directory"]}
export PYTHONPATH="{metadata["initial_pythonpath"]}"
python3 {metadata["script_path"]} {" ".join(content if content[0] != "[]" else [])} {" ".join(list(filter(lambda x: "--" in x and not "multirun" in x, sys.argv)))} </code></pre>
            </main>
            """
            + """
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
            images.append(f'<img src="gfx/{filename}" style="object-fit: cover; width: 100%; max-height: 100%;">')
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


class HistoricalCallback(Callback, ABC):
    def __init__(self, *args, **kwargs):
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
                assert idx >= 1, f"Indices should be no smaller than 1."
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

    def plot_gui(self):
        self.__data = self.load_data(idx="last")
        if not self.data.empty:
            return self.plot(name=self.__class__.__name__)
        else:
            return None


def method_callback(method_name, class_name=None, log_level="debug"):
    """
    Creates a callback class that logs the output of a specific method of a class or any class.

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
        def __init__(self, log, log_level=log_level):
            super().__init__(log, log_level=log_level)

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
    """
    StateCallback is a subclass of the Callback class that logs the state of a system object when the compute_closed_loop_rhs method of the System class is called.

    Attributes:
    log (function): A function that logs a message at a specified log level.
    """

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, rcognita.systems.System)
            and method == rcognita.systems.System.compute_closed_loop_rhs.__name__
        )

    def perform(self, obj, method, output):
        self.log(f"System's state: {obj._state}")


class ObjectiveCallback(Callback):
    cooldown = 8.0
    """
    A Callback class that logs the current objective value of an Actor instance.

    This callback is triggered whenever the Actor.objective method is called.

    Attributes:
    log (function): A logger function with the specified log level.
    """

    def is_target_event(self, obj, method, output):
        return isinstance(obj, rcognita.actors.Actor) and method == "objective"

    def perform(self, obj, method, output):
        self.log(f"Current objective: {output}")


class HistoricalObjectiveCallback(HistoricalCallback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.timeline = []
        self.num_launch = 1
        self.counter = 0
        self.cooldown = 1.0

    def on_launch(self):
        super().on_launch()
        with rcognita.main.metadata["report"]() as r:
            r["elapsed_relative"] = 0

    def is_target_event(self, obj, method, output):
        return isinstance(obj, rcognita.scenarios.Scenario) and (method == "post_step")

    def perform(self, obj, method, output):
        self.counter += 1
        self.log(
            f"Current objective: {output[0]}, observation: {output[1]}, action: {output[2]}, total objective: {output[3]:.4f}, time: {obj.time:.4f} ({100 * obj.time/obj.simulator.time_final:.1f}%), episode: {obj.episode_counter + 1}/{obj.N_episodes}"
        )
        if not self.counter % 3:
            do_exit = False
            with rcognita.main.metadata["report"]() as r:
                r["elapsed_relative"] = obj.time / obj.simulator.time_final
                if "terminate" in r:
                    do_exit = True
            if do_exit:
                self.log("Termination request issued from GUI.")
                raise rcognita.RcognitaExitException(
                    "Termination request issued from gui."
                )

        self.add_datum(
            {
                "time": round(output[3], 4),
                "current objective": output[0],
                "observation": output[1],
                "action": output[2],
                "total_objective": round(output[3], 4),
                "completed_percent": 100
                * round(obj.time / obj.simulator.time_final, 1),
            }
        )

    def on_episode_done(self, scenario, episode_number, episodes_total):
        self.insert_column_left("episode", episode_number)
        self.dump_and_clear_data(f"episode_{str(episode_number).zfill(5)}")


class SaveProgressCallback(Callback):
    once_in = 1

    def on_launch(self):
        with rcognita.main.metadata["report"]() as r:
            r["episode_current"] = 0

    def is_target_event(self, obj, method, output):
        return False

    def perform(self, obj, method, output):
        pass

    def on_episode_done(self, scenario, episode_number, episodes_total):
        start = time.time()
        if episode_number % self.once_in:
            return
        filename = f"callbacks.dill"
        with open(filename, "wb") as f:
            dill.dump(rcognita.main.callbacks, f)
        self.log(
            f"Saved callbacks to {os.path.abspath(filename)}. ({int(1000 * (time.time() - start))}ms)"
        )
        if scenario is not None:
            with rcognita.main.metadata["report"]() as r:
                r["episode_current"] = episode_number
                if "episodes_total" not in r:
                    r["episode_total"] = episodes_total
                r["elapsed_relative"] = 1.0

    def on_termination(self, res):
        if isinstance(res, Exception):
            self.on_episode_done(None, 0, None)


class HistoricalObservationCallback(HistoricalCallback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cooldown = 0.0

        self.current_episode = None
        self.dt_simulator_counter = 0
        self.time_threshold = 10

    def is_target_event(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and (method == "post_step"):
            self.dt_simulator_counter += 1
            return not self.dt_simulator_counter % self.time_threshold

    def perform(self, obj, method, output):
        self.add_datum(
            {
                **{
                    "time": obj.time,
                    "action": obj.action,
                },
                **dict(zip(obj.observation_components_naming, obj.observation)),
            }
        )

    def on_episode_done(self, scenario, episode_number, episodes_total):
        identifier = f"observations_in_episode_{str(episode_number).zfill(5)}"
        self.save_plot(identifier)
        self.insert_column_left("episode", episode_number)
        self.dump_and_clear_data(identifier)

    def plot(self, name=None):
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


class TotalObjectiveCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = pd.DataFrame()
        self.xlabel = "Episode"
        self.ylabel = "Total objective"

    def is_target_event(self, obj, method, output):
        return isinstance(obj, rcognita.scenarios.Scenario) and (
            method == "reload_pipeline"
        )

    def perform(self, obj, method, output):
        self.add_datum({"episode": len(self.data) + 1, "objective": output})
        self.log(
            f"Final total objective of episode {self.data.iloc[-1]['episode']} is {round(self.data.iloc[-1]['objective'], 2)}"
        )
        self.dump_data("Total_Objective")
        self.save_plot("Total_Objective")

    def load_data(self, idx=None):
        return super().load_data(idx=1)

    def plot(self, name=None):
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


class QFunctionModelSaverCallback(HistoricalCallback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cooldown = 0.0

    def plot_gui(self):
        return None

    def load_data(self, idx=None):
        assert idx is not None, "Provide idx=..."
        return torch.load(
            os.path.join(
                self.get_save_directory(),
                f"critic_model_{str(idx).zfill(5)}.pt",
            )
        )

    def is_target_event(self, obj, method, output):
        return False

    def perform(self, obj, method, output):
        pass

    def on_episode_done(self, scenario, episode_number, episodes_total):
        if "CriticOffPolicy" in scenario.critic.__class__.__name__:
            torch.save(
                scenario.critic.model.state_dict(),
                f"{self.get_save_directory()}/critic_model_{str(episode_number).zfill(5)}.pt",
            )


class QFunctionCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cooldown = 0.0

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, rcognita.scenarios.Scenario)
            and method == "post_step"
            and "CriticOffPolicy" in obj.critic.__class__.__name__
        )

    def perform(self, obj, method, output):
        self.add_datum(
            {
                **{
                    "time": obj.time,
                    "Q-Function-Value": obj.critic.model(obj.observation, obj.action)
                    .detach()
                    .cpu()
                    .numpy(),
                    "action": obj.action,
                },
                **dict(zip(obj.observation_components_naming, obj.observation)),
            }
        )

    def on_episode_done(self, scenario, episode_number, episodes_total):
        self.insert_column_left("episode", episode_number)
        self.dump_and_clear_data(f"q_function_values_{str(episode_number).zfill(5)}")


class TimeRemainingCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_episode = []

    def is_target_event(self, obj, method, output):
        return False

    def perform(self, obj, method, output):
        pass

    def on_episode_done(self, scenario, episode_number, episodes_total):
        self.time_episode.append(time.time())
        if len(self.time_episode) > 3:
            average_interval = 0
            previous = self.time_episode[-1]
            for current in self.time_episode[-2:-12:-1]:
                average_interval -= current - previous
                previous = current
            average_interval /= len(self.time_episode[-2:-12:-1])
            td = datetime.timedelta(
                seconds=int(average_interval * (episodes_total - episode_number))
            )
            remaining = f" Estimated remaining time: {str(td)}."
        else:
            remaining = ""
        self.log(
            f"Completed episode {episode_number}/{episodes_total} ({100*episode_number/episodes_total:.1f}%)."
            + remaining
        )


class CriticObjectiveCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cooldown = 1.0
        self.time = 0.0

    def is_target_event(self, obj, method, output):
        return isinstance(obj, rcognita.critics.Critic) and method == "objective"

    def perform(self, obj, method, output):
        self.log(f"Current TD value: {output}")
        self.add_datum(
            {
                "time": rcognita.main.metadata["time"],
                "TD value": output.detach().cpu().numpy(),
            }
        )

    def on_episode_done(self, scenario, episode_number, episodes_total):
        self.insert_column_left("episode", episode_number)
        self.dump_and_clear_data(f"TD_values_{str(episode_number).zfill(5)}")


class CalfCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cooldown = 1.0
        self.time = 0.0

    def is_target_event(self, obj, method, output):
        return (
            isinstance(obj, rcognita.controllers.Controller)
            and method == "compute_action"
            and "calf" in obj.critic.__class__.__name__.lower()
        )

    def perform(self, obj, method, output):
        current_CALF = obj.critic(
            obj.critic.observation_last_good, use_stored_weights=True
        )
        self.log(
            f"current CALF value:{current_CALF}, decay_rate:{obj.critic.safe_decay_rate}, observation: {obj.critic.observation_buffer[:,-1]}"
        )
        is_calf = (
            obj.critic.weights_acceptance_status == "accepted"
            and obj.actor.weights_acceptance_status == "accepted"
        )
        if not self.data.empty:
            prev_CALF = self.data["J_hat"].iloc[-1]
        else:
            prev_CALF = current_CALF

        delta_CALF = prev_CALF - current_CALF
        self.add_datum(
            {
                "time": rcognita.main.metadata["time"],
                "J_hat": current_CALF,
                "is_CALF": is_calf,
                "delta": delta_CALF,
            }
        )

    def on_episode_done(self, scenario, episode_number, episodes_total):
        identifier = f"CALF_diagnostics_on_episode_{str(episode_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", episode_number)
            self.dump_and_clear_data(identifier)

    def plot(self, name=None):
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
