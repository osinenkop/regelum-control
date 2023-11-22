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
        return isinstance(obj, regelum.scenario.Scenario) and method == "compute_action"

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
        return isinstance(obj, regelum.scenario.Scenario) and method == "compute_action"

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
