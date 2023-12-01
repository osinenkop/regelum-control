import regelum
from abc import abstractmethod, ABC
import numpy as np
from regelum.data_buffers import DataBuffer
from typing import Union


class StabilizationCriterion(regelum.RegelumBase, ABC):
    @abstractmethod
    def __call__(self, data_buffer: DataBuffer) -> bool:
        pass


class CoordinateMaxAbsCriterion(StabilizationCriterion):
    def __init__(self, max_abs: Union[float, np.ndarray], n_last_observations: int):
        self.max_abs = max_abs
        self.n_latest_observations = n_last_observations

    def __call__(self, data_buffer: DataBuffer) -> bool:
        data = data_buffer.to_pandas(["observation", "episode_id"])
        data["episode_id"] = data["episode_id"].astype(int)
        latest = (
            data.groupby(["episode_id"])["observation"]
            .apply(lambda x: x[-self.n_latest_observations :])
            .groupby(level=0)
            .mean()
        )

        for v in latest:
            if np.any(np.array(np.abs(v), dtype=float) >= self.max_abs):
                return False
        return True


class InvertedPendulumCriterion(CoordinateMaxAbsCriterion):
    def __init__(self):
        super().__init__(np.array([0.1, 0.1]), n_last_observations=10)


class TwoTankCriterion(CoordinateMaxAbsCriterion):
    def __init__(self):
        super().__init__(max_abs=0.1, n_last_observations=10)


class ThreeWheeledRobotKinematicCriterion(CoordinateMaxAbsCriterion):
    def __init__(self):
        super().__init__(max_abs=0.3, n_last_observations=10)


class ThreeWheeledRobotDynamicCriterion(CoordinateMaxAbsCriterion):
    def __init__(self):
        super().__init__(max_abs=0.3, n_last_observations=10)


class LunarLanderCriterion(CoordinateMaxAbsCriterion):
    def __init__(self):
        super().__init__(
            max_abs=np.array([0.2, 0.01, 0.001, np.inf, np.inf, np.inf]),
            n_last_observations=10,
        )


class KinPointCriterion(CoordinateMaxAbsCriterion):
    def __init__(self):
        super().__init__(
            max_abs=0.05,
            n_last_observations=10,
        )


class CartpoleCriterion(CoordinateMaxAbsCriterion):
    def __init__(self):
        super().__init__(max_abs=0.05, n_last_observations=10)


class NeverCriterion(StabilizationCriterion):
    def __call__(self, data_buffer: DataBuffer) -> bool:
        return False
