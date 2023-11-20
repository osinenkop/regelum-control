class TimeCallback(Callback):
    """Callback responsible for keeping track of simulation time."""

    def is_target_event(self, obj, method, output):
        return isinstance(obj, regelum.scenario.Scenario) and method == "post_step"

    def on_launch(self):
        regelum.main.metadata["time"] = 0.0

    def perform(self, obj, method, output):
        regelum.main.metadata["time"] = obj.time if not obj.is_episode_ended else 0.0
