import numpy as np


class SignalGenerator:
    def __init__(self, signals_paths: dict):
        self.signals_paths = signals_paths

    def __call__(self, *args, **kwargs):
        return self.get_signal(*args, **kwargs)

    def get_signal(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class SinusoidalExample(SignalGenerator):
    def get_signal(self, t) -> dict:
        return {"w": np.sin(t)}
