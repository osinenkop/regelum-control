from regelum.policy import Policy


class SimplePolicy(Policy):
    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain

    def get_action(self, observation):
        return -self.gain * observation
