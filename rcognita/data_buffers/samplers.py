from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from .types import ArrayType, Array
import numpy as np


class Sampler(ABC):
    def __init__(
        self,
        data_buffer=None,
        keys: Optional[List[str]] = None,
        dtype: ArrayType = np.array,
    ):
        self.keys = keys
        self.dtype = dtype

        if data_buffer is not None:
            self.attach_data_buffer(data_buffer)

        self.wrapper_on = False

    def __iter__(self):
        if self.stop_iteration_criterion():
            self.nullify_sampler()
        return self

    def __next__(self):
        if self.stop_iteration_criterion():
            raise StopIteration
        return self.next()

    @abstractmethod
    def next(self) -> Dict[str, Array]:
        pass

    @abstractmethod
    def nullify_sampler(self) -> None:
        pass

    @abstractmethod
    def stop_iteration_criterion(self) -> bool:
        pass

    def attach_data_buffer(self, data_buffer) -> None:
        self.data_buffer = data_buffer
        self.dtype = (
            self.dtype if self.dtype is not None else data_buffer.dtype_for_indexing
        )
        self.keys = (
            self.keys if self.keys is not None else data_buffer.keys_for_indexing
        )

        self.data_buffer.set_indexing_rules(keys=self.keys, dtype=self.dtype)


class ForwardSampler(Sampler):
    def __init__(
        self,
        *args,
        batch_size: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.nullify_sampler()

    def nullify_sampler(self) -> None:
        self.batch_ids = np.arange(self.batch_size, dtype=int) - 1

    def stop_iteration_criterion(self) -> bool:
        return self.batch_ids[-1] >= len(self.data_buffer) - 1

    def next(self) -> Dict[str, Array]:
        self.batch_ids += 1
        return self.data_buffer[self.batch_ids]


class EpisodicSampler(Sampler):
    def __init__(
        self,
        *args,
        batch_size,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.nullify_sampler()

    def nullify_sampler(self) -> None:
        self.episode_ids = self.data_buffer.to_pandas(
            keys=["episode_id"]
        ).values.reshape(-1)
        self.max_episode_id = max(self.episode_ids)
        self.cur_episode_id = min(self.episode_ids) - 1
        self.idx_batch = -1

    def stop_iteration_criterion(self) -> bool:
        return self.cur_episode_id >= self.max_episode_id

    def get_episode_batch_ids(self, episode_id) -> np.array:
        return np.arange(len(self.data_buffer), dtype=int)[
            self.episode_ids == episode_id
        ]

    def next(self) -> Dict[str, Array]:
        # self.idx_batch += 1
        # batch_ids = self.get_episode_batch_ids(self.cur_episode_id)
        # if self.idx_batch * self.batch_size >= len(batch_ids) - 1:
        #     self.cur_episode_id += 1
        #     self.idx_batch = 0
        #     batch_ids = self.get_episode_batch_ids(self.cur_episode_id)

        # return self.data_buffer.getitem(
        #     batch_ids[
        #         self.idx_batch
        #         * self.batch_size : (self.idx_batch + 1)
        #         * self.batch_size
        #     ],
        #     self.keys,
        #     self.dtype,
        # )
        self.cur_episode_id += 1
        batch_ids = self.get_episode_batch_ids(self.cur_episode_id)
        return self.data_buffer[batch_ids]
