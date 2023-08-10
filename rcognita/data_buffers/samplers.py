from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union
from .types import ArrayType, Array
import numpy as np
import torch


class Sampler(ABC):
    def __init__(
        self,
        data_buffer,
        keys: Optional[List[str]],
        dtype: ArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.keys = keys
        self.dtype = dtype
        self.data_buffer = data_buffer
        self.data_buffer.set_indexing_rules(
            keys=self.keys, dtype=self.dtype, device=device
        )
        self.len_data_buffer = len(self.data_buffer.data[self.keys[0]])
        self.device = device
        for k in self.keys:
            assert self.len_data_buffer == len(
                self.data_buffer.data[k]
            ), "All keys should have the same length in Data Buffer"

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


class RollingSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        mode: str,
        data_buffer,
        keys: Optional[List[str]],
        n_batches: Optional[int] = None,
        dtype: ArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
    ):
        assert batch_size > 0, "Batch size should be > 0"
        assert mode in [
            "uniform",
            "backward",
            "forward",
        ], "mode should be one of ['uniform', 'backward', 'forward']"
        assert not (
            n_batches is None and mode == "uniform"
        ), "'uniform' mode is not avaliable for n_batches == None"

        Sampler.__init__(
            self, data_buffer=data_buffer, keys=keys, dtype=dtype, device=device
        )
        self.mode = mode
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_batches_sampled: int
        self.nullify_sampler()

    def nullify_sampler(self) -> None:
        self.n_batches_sampled = 0
        if self.mode == "forward":
            self.batch_ids = np.arange(self.batch_size, dtype=int)
        elif self.mode == "backward":
            self.batch_ids = np.arange(
                self.len_data_buffer - self.batch_size,
                self.len_data_buffer,
                dtype=int,
            )
        elif self.mode == "uniform":
            self.batch_ids = np.random.randint(
                low=0,
                high=max(self.len_data_buffer - self.batch_size, 1),
            ) + np.arange(self.batch_size, dtype=int)
        else:
            raise ValueError("mode should be one of ['uniform', 'backward', 'forward']")

    def stop_iteration_criterion(self) -> bool:
        if self.len_data_buffer <= self.batch_size:
            return True
        if self.mode == "forward":
            return (
                self.batch_ids[-1] >= len(self.data_buffer)
                or self.n_batches == self.n_batches_sampled
            )
        elif self.mode == "backward":
            return self.batch_ids[0] <= 0 or self.n_batches == self.n_batches_sampled
        elif self.mode == "uniform":
            return self.n_batches == self.n_batches_sampled
        else:
            raise ValueError("mode should be one of ['uniform', 'backward', 'forward']")

    def next(self) -> Dict[str, Array]:
        batch = self.data_buffer[self.batch_ids]
        if self.mode == "forward":
            self.batch_ids += 1
        elif self.mode == "backward":
            self.batch_ids -= 1
        elif self.mode == "uniform":
            self.batch_ids = np.random.randint(
                low=0, high=self.len_data_buffer - self.batch_size
            ) + np.arange(self.batch_size, dtype=int)
        else:
            raise ValueError("mode should be one of ['uniform', 'backward', 'forward']")

        self.n_batches_sampled += 1
        return batch


class EpisodicSampler(Sampler):
    def __init__(
        self,
        data_buffer=None,
        keys: Optional[List[str]] = None,
        dtype: ArrayType = np.array,
        device: Optional[Union[str, torch.device]] = None,
    ):
        Sampler.__init__(
            self, data_buffer=data_buffer, keys=keys, dtype=dtype, device=device
        )
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
