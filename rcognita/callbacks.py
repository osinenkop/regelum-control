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

import rcognita
import pandas as pd
import time, datetime
import dill, os

import omegaconf, json

import matplotlib.pyplot as plt
import re

import pkg_resources


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
        name = metadata["config_path"].split("/")[-1].split(".")[0]
        cfg.treemap(root=name).write_html("SUMMARY.html")
        with open("SUMMARY.html", "r") as f:
            html = f.read()
        try:
            stream = os.popen("git log --name-status HEAD^..HEAD")
            commit_hash = stream.read().split()[1]
            stream = os.popen("git status").read()
            if not "is up to date" in stream or "not staged" in stream:
                commit_hash += (
                    ' <font color="red">(uncommitted/unstaged changes)</font>'
                )
                if "disallow_uncommitted" in cfg and cfg.disallow_uncommitted:
                    raise Exception(
                        "Running experiments without committing is disallowed. Please, commit your changes."
                    )
        except:
            commit_hash = None
            if "disallow_uncommitted" in cfg and cfg.disallow_uncommitted:
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

        self.episodic_cache = {}
        self.cache = {}
        self.timeline = []
        self.num_launch = 1
        self.cooldown = 0.0
        self.current_episode = None
        self.columns = None

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "post_step":
            if self.columns is None:
                self.columns = obj.observation_components_naming
            key = obj.time

            self.episodic_cache[key] = output[1]
            self.current_episode = obj.episode_counter
        elif (
            isinstance(obj, rcognita.scenarios.Scenario) and method == "reload_pipeline"
        ):
            self.cache[self.current_episode] = self.episodic_cache
            self.save_plot(
                f"observations_in_episode_{str(obj.episode_counter).zfill(5)}"
            )
            self.episodic_cache = {}

    @property
    def data(self):
        return self.cache

    @property
    def episodic_data(self):
        return pd.DataFrame.from_dict(
            self.episodic_cache, orient="index", columns=self.columns
        )

    @classmethod
    def name_observation_components(cls, columns):
        cls.columns = columns

    def plot(self, name=None):
        if not name:
            name = self.__class__.__name__
        self.episodic_data.plot(subplots=True, grid=True, xlabel="time", title=name)
        # plt.xticks(range(1, len(self.data) + 1))


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


class QFunctionCallback(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache = {}
        self.episodic_cache = {}
        self.timeline = []
        self.num_launch = 1
        self.cooldown = 0.0
        self.current_episode = None

    def perform(self, obj, method, output):
        if (
            isinstance(obj, rcognita.scenarios.Scenario)
            and method == "post_step"
            and obj.critic.__class__.__name__ == "CriticOffPolicy"
        ):
            key = obj.time
            self.episodic_cache[key] = {
                "observation": obj.observation,
                "action": obj.action,
                "Q-Function-Value": obj.critic.model(obj.observation, obj.action)
                .detach()
                .cpu()
                .numpy(),
            }
            self.current_episode = obj.episode_counter
        elif (
            isinstance(obj, rcognita.scenarios.Scenario)
            and method == "reload_pipeline"
            and obj.critic.__class__.__name__ == "CriticOffPolicy"
        ):
            self.cache[self.current_episode] = self.episodic_cache


class SaveProgressCallback(Callback):
    once_in = 1

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.scenarios.Scenario) and method == "reload_pipeline":
            start = time.time()
            episode = obj.episode_counter
            if episode % self.once_in:
                return
            filename = f"callbacks_at_episode_{episode + 1}.dill"
            prev_filename = f"callbacks_at_episode_{episode + 1 - self.once_in}.dill"
            with open(filename, "wb") as f:
                dill.dump(rcognita.main.callbacks, f)
            if episode > self.once_in:
                os.remove(os.path.abspath(prev_filename))
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


class CriticObjectiveCallback(Callback):
    cooldown = 1

    def perform(self, obj, method, output):
        if isinstance(obj, rcognita.critics.Critic) and method == "objective":
            self.log(f"Current TD value: {output}")


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

        ax_calf, ax_switch, ax_delta = fig.subplots(1, 3)
        ax_calf.plot(self.data.iloc[:, 1], label="CALF")
        ax_switch.plot(self.data.iloc[:, 2], label="CALF persistency")
        ax_delta.plot(self.data.iloc[:, 3], label="delta decay")

        plt.grid()
        plt.legend()
