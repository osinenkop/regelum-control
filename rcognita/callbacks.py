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
import dill, os, git

import omegaconf, json

import matplotlib.pyplot as plt
import re

import pkg_resources

import sys


def is_in_debug_mode():
<<<<<<< HEAD
    return not getattr(sys, "gettrace", None) is None
=======
    return not sys.gettrace() is None
>>>>>>> 086c847a82de2fe103228ffdd9de4e4f839826b1


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

    def ready(self):
        if not self.cooldown:
            return True
        if time.time() - self.last_trigger > self.cooldown:
            self.last_trigger = time.time()
            return True
        else:
            return False

    def on_launch(self, cfg, metadata):
        pass

    @abstractmethod
    def perform(self, obj, method, output):
        pass

    def __call__(self, obj, method, output):
        if not self.ready():
            return
        self.performed_bases = []
        for base in self.__class__.__bases__:
            if ABC not in base.__bases__ and base not in self.peformed_bases:
                base(self, obj, method, output)
                self.performed_bases.append(base)
        try:
            self.perform(obj, method, output)
        except Exception as e:
            self.log(f"Callback {self.__class__.__name__} failed.")
            self.exception(e)

    def on_termination(self):
        pass


class ConfigDiagramCallback(Callback):
    def perform(self, *args, **kwargs):
        pass

    def on_launch(self, cfg, metadata):
        start = time.time()
        os.mkdir(".summary")
        name = metadata["config_path"].split("/")[-1].split(".")[0]
        cfg.treemap(root=name).write_html("SUMMARY.html")
        with open("SUMMARY.html", "r") as f:
            html = f.read()
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha
            if repo.is_dirty(untracked_files=True):
                commit_hash += (
                    ' <font color="red">(uncommitted/unstaged changes)</font>'
                )
                if (
                    "disallow_uncommitted" in cfg
                    and cfg.disallow_uncommitted
                    and not is_in_debug_mode()
                ):
                    raise Exception(
                        "Running experiments without committing is disallowed. Please, commit your changes."
                    )
                untracked = repo.untracked_files
                repo.git.add(all=True)
                diff = repo.git.diff(repo.head.commit.tree)
                if untracked:
                    repo.index.remove(untracked, cached=True)
                with open(".summary/changes.diff", "w") as f:
                    f.write(diff + "\n")
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
                for line in content:
                    field, value = line.split("=")
                    overrides_table += f'<tr><td><font face="Courier New">{field}</font></td> <td><font face="Courier New"> = </font></td>  <td><font face="Courier New">{value}</font></td> </tr>\n'
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
                            <div style="float: left; width: 50%">
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
            """
            + "<br>" * (max(len(content), 5) + 1),
        )
        html = html.replace(
            "<body>",
            f"""
                            <body>
                            <div style="float: right; width: 50%">
                            <table>
                            <tbody>
                            {overrides_table}
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
git restore .
git clean -f
{f'''git checkout {commit_hash.replace(' <font color="red">(uncommitted/unstaged changes)</font>',  chr(10) + f'patch -p1 < {os.path.abspath(".summary/changes.diff")}')}''' + chr(10) if commit_hash else ""}cd {metadata["initial_working_directory"]}
export PYTHONPATH="{metadata["initial_pythonpath"]}"
python3 {metadata["script_path"]} {" ".join(content)} {" ".join(list(filter(lambda x: "--" in x and not "multirun" in x, sys.argv)))} </code></pre>
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

    def on_termination(self):
        with open("SUMMARY.html", "r") as f:
            html = f.read()
        table_lines = ""
        images = []
        directory = os.fsencode("gfx")

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            images.append(f'<img src="gfx/{filename}">')
        if len(images) % 2:
            images.append("")
        for i in range(0, len(images), 2):
            table_lines += f"<tr><td>{images[i]}</td> <td>{images[i + 1]}</td></tr>\n"
        html = html.replace(
            "</body>",
            f"""
                            <br>
                            <table>
                            <tbody>
                            {table_lines}
                            </tbody>
                            </table>
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

    @property
    @abstractmethod
    def data(self):
        pass

    def plot(self, name=None):
        if not name:
            name = self.__class__.__name__
        self.data.plot()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(name)
        plt.grid()
        # plt.xticks(range(1, len(self.data) + 1))

    def save_plot(self, name=None):
        if not name:
            name = self.__class__.__name__
        self.plot(name=name)
        plt.savefig(f"gfx/{name}.png")


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

        def perform(self, obj, method, output):
            if method == method_name and class_name in [None, obj.__class__.__name__]:
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

    def perform(self, obj, method, output):
        if (
            isinstance(obj, rcognita.systems.System)
            and method == rcognita.systems.System.compute_closed_loop_rhs.__name__
        ):
            self.log(f"System's state: {obj._state}")


class ObjectiveCallback(Callback):
    """
    A Callback class that logs the current objective value of an Actor instance.

    This callback is triggered whenever the Actor.objective method is called.

    Attributes:
    log (function): A logger function with the specified log level.
    """

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.actors.Actor) and method == "objective":
            self.log(f"Current objective: {output}")


class HistoricalObjectiveCallback(HistoricalCallback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache = {}
        self.timeline = []
        self.num_launch = 1
        self.cooldown = 0.0

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "post_step":
            self.log(
                f"Current objective: {output[0]}, observation: {output[1]}, action: {output[2]}, total objective: {output[3]:.4f}, time: {obj.time:.4f} ({100 * obj.time/obj.simulator.time_final:.1f}%)"
            )
            key = (self.num_launch, obj.time)
            if key in self.cache.keys():
                self.num_launch += 1
                key = (self.num_launch, obj.time)

            self.cache[key] = output[0]
            if self.timeline != []:
                if self.timeline[-1] < key[1]:
                    self.timeline.append(key[1])

            else:
                self.timeline.append(key[1])

    @property
    def data(self):
        keys = list(self.cache.keys())
        run_numbers = sorted(list(set([k[0] for k in keys])))
        cache_transformed = {key: list() for key in run_numbers}
        for k, v in self.cache.items():
            cache_transformed[k[0]].append(v)
        return cache_transformed


class HistoricalObservationCallback(HistoricalCallback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.episodic_cache = []
        self.cache = []
        self.timeline = []
        self.num_launch = 1
        self.cooldown = 0.0
        self.current_episode = None

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "post_step":
            self.current_episode = obj.episode_counter + 1
            self.episodic_cache.append(
                {
                    **{"episode": self.current_episode, "time": obj.time},
                    **dict(zip(obj.observation_components_naming, output[1])),
                }
            )
        elif (
            isinstance(obj, rcognita.scenarios.Scenario) and method == "reload_pipeline"
        ):
            self.cache += self.episodic_cache
            self.save_plot(
                f"observations_in_episode_{str(self.current_episode).zfill(5)}"
            )
            self.episodic_cache = []

    @property
    def data(self):
        df = pd.DataFrame.from_records(self.cache)
        df.set_index(["episode", "time"], inplace=True)
        return df

    @property
    def episodic_data(self):
        df = pd.DataFrame.from_records(self.episodic_cache)
        df.drop(["episode"], axis=1, inplace=True)
        df.set_index("time", inplace=True)

        return df

    @classmethod
    def name_observation_components(cls, columns):
        cls.columns = columns

    def plot(self, name=None):
        if not name:
            name = self.__class__.__name__
        self.episodic_data.plot(subplots=True, grid=True, xlabel="time", title=name)


class TotalObjectiveCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = pd.DataFrame()

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "reload_pipeline":
            self.log(f"Current total objective: {output}")
            episode = (
                obj.episode_counter
                if obj.episode_counter != 0
                else self.cache.index.max() + 1
            )
            row = pd.DataFrame({"objective": output}, index=[episode])
            self.cache = pd.concat([self.cache, row])
            self.save_plot("Total objective")

    @property
    def data(self):
        return self.cache


class QFunctionModelSaverCallback(Callback):
    """
    A callback which allows to store desired data
    collected among different runs inside multirun execution runtime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.timeline = []
        self.num_launch = 1
        self.cooldown = 0.0
        self.current_episode = None

        os.mkdir("checkpoints")

    def perform(self, obj, method, output):
        if (
            isinstance(obj, rcognita.scenarios.Scenario)
            and method == "post_step"
            and obj.critic.__class__.__name__ == "CriticOffPolicy"
        ):
            self.current_episode = obj.episode_counter + 1
            torch.save(
                obj.critic.model.state_dict(),
                f"checkpoints/critic_model_{str(self.current_episode).zfill(5)}_{round(obj.time, 2)}.pt",
            )
        elif (
            isinstance(obj, rcognita.scenarios.Scenario)
            and method == "reload_pipeline"
            and obj.critic.__class__.__name__ == "CriticOffPolicy"
        ):
            pass
            # torch.save(
            #     obj.critic.model.state_dict(),
            #     f"checkpoints/critic_model_{str(self.current_episode).zfill(5)}.pt",
            # )


class QFunctionCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache = {"pre_step": {}, "post_step": {}}
        self.pre_step_episodic_cache = {}
        self.post_step_episodic_cache = {}
        self.timeline = []
        self.num_launch = 1
        self.cooldown = 0.0
        self.current_episode = None

    def perform(self, obj, method, output):
        if (
            isinstance(obj, rcognita.scenarios.Scenario)
            and method == "pre_step"
            and obj.critic.__class__.__name__ == "CriticOffPolicy"
        ):
            key = obj.time
            self.current_episode = obj.episode_counter + 1
            self.pre_step_episodic_cache[key] = {
                "observation": obj.observation,
                "action": obj.action,
                "Q-Function-Value": obj.critic.model(obj.observation, obj.action)
                .detach()
                .cpu()
                .numpy(),
            }
        elif (
            isinstance(obj, rcognita.scenarios.Scenario)
            and method == "post_step"
            and obj.critic.__class__.__name__ == "CriticOffPolicy"
        ):
            key = obj.time
            self.current_episode = obj.episode_counter + 1
            self.post_step_episodic_cache[key] = {
                "observation": obj.observation,
                "action": obj.action,
                "Q-Function-Value": obj.critic.model(obj.observation, obj.action)
                .detach()
                .cpu()
                .numpy(),
            }
        elif (
            isinstance(obj, rcognita.scenarios.Scenario)
            and method == "reload_pipeline"
            and obj.critic.__class__.__name__ == "CriticOffPolicy"
        ):
            self.cache["post_step"][
                self.current_episode
            ] = self.post_step_episodic_cache
            self.cache["pre_step"][self.current_episode] = self.pre_step_episodic_cache
            self.post_step_episodic_cache = {}
            self.pre_step_episodic_cache = {}

    @property
    def data(self):
        return self.cache


class SaveProgressCallback(Callback):
    once_in = 1

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "reload_pipeline":
            start = time.time()
            episode = obj.episode_counter
            if episode % self.once_in:
                return
            filename = f"callbacks.dill"
            with open(filename, "wb") as f:
                dill.dump(rcognita.main.callbacks, f)
            self.log(
                f"Saved callbacks to {os.path.abspath(filename)}. ({int(1000 * (time.time() - start))}ms)"
            )


class TimeRemainingCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_episode = []

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "reload_pipeline":
            self.time_episode.append(time.time())
            total_episodes = obj.N_episodes
            current_episode = (
                obj.episode_counter if obj.episode_counter != 0 else obj.N_episodes
            )
            if len(self.time_episode) > 3:
                average_interval = 0
                previous = self.time_episode[-1]
                for current in self.time_episode[-2:-12:-1]:
                    average_interval -= current - previous
                    previous = current
                average_interval /= len(self.time_episode[-2:-12:-1])
                td = datetime.timedelta(
                    seconds=int(average_interval * (total_episodes - current_episode))
                )
                remaining = f" Estimated remaining time: {str(td)}."
            else:
                remaining = ""
            self.log(
                f"Completed episode {current_episode}/{total_episodes} ({100*current_episode/total_episodes:.1f}%)."
                + remaining
            )


class CriticObjectiveCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache = []
        self.timeline = []
        self.num_launch = 1
        self.cooldown = 0.0
        self.current_episode = None

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.critics.Critic) and method == "objective":
            self.log(f"Current TD value: {output}")
            self.cache.append(output)

    @property
    def data(self):
        return self.cache


class CalfCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = pd.DataFrame()

    def perform(self, obj, method, output):
        if (
            isinstance(obj, rcognita.controllers.Controller)
            and method == "compute_action"
        ):
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
            if not self.cache.empty:
                CALF_prev = self.cache["J_hat"].iloc[-1]
            else:
                CALF_prev = current_CALF

            delta_CALF = CALF_prev - current_CALF
            row = pd.DataFrame(
                {
                    "J_hat": [current_CALF],
                    "is_CALF": [is_calf],
                    # "weights": [current_weights],
                    "delta": [delta_CALF],
                }
            )
            self.cache = pd.concat([self.cache, row], axis=0)

    @property
    def data(self):
        return self.cache

    def plot(self):
        self.data.reset_index(inplace=True)

        fig = plt.figure(figsize=(10, 10))

        ax_calf, ax_switch, ax_delta = ax_array = fig.subplots(1, 3)
        ax_calf.plot(self.data.iloc[:, 1], label="CALF")
        ax_switch.plot(self.data.iloc[:, 2], label="CALF on")
        ax_delta.plot(self.data.iloc[:, 3], label="delta decay")

        for ax in ax_array:
            ax.set_xlabel("Time [s]")
            ax.grid()
            ax.legend()

        ax_calf.set_ylabel("J_hat")
        ax_switch.set_ylabel("Is CALF on")
        ax_delta.set_ylabel("Decay size")

        plt.legend()
