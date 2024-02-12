"""Contains BatchSamplers for data buffers."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union
from regelum.typing import RgArrayType, RgArray
import numpy as np
import torch


class BatchSampler(ABC):
    """Base class for batch samplers."""

    def __init__(
        self,
        data_buffer,
        keys: Optional[List[str]],
        dtype: RgArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ):
        """Instantiate a BatchSampler.

        Args:
            data_buffer (DataBuffer): Data Buffer instance
            keys (Optional[List[str]]): keys to sample
            dtype (RgArrayType, optional): dtype for sample, can be
                either cs.DM, np.array, torch.Tensor, defaults to
                torch.FloatTensor
            device (Optional[Union[str, torch.device]], optional):
                device for sampling, needed for torch.FloatTensor
                defaults to None
            fill_na (Optional[float], optional, defaults to 0.0): fill
                value for np.nan, defaults to 0.0
        """
        self.keys = keys
        self.dtype = dtype
        self.data_buffer = data_buffer
        self.data_buffer.set_indexing_rules(
            keys=self.keys, dtype=self.dtype, device=device, fill_na=fill_na
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
    def next(self) -> Dict[str, RgArray]:
        pass

    @abstractmethod
    def nullify_sampler(self) -> None:
        pass

    @abstractmethod
    def stop_iteration_criterion(self) -> bool:
        pass


class RollingBatchSampler(BatchSampler):
    """Batch sampler for rolling batches."""

    def __init__(
        self,
        mode: str,
        data_buffer,
        keys: Optional[List[str]],
        batch_size: Optional[int] = None,
        n_batches: Optional[int] = None,
        dtype: RgArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ):
        """Instantiate a RollingBatchSampler.

        Args:
            mode (str): mode for batch sampling. Can be either
                'uniform', 'backward', 'forward', 'full'. 'forward' for
                sampling of rolling batches from the beginning of
                DataBuffer. 'backward' for sampling of rolling batches
                from the end of DataBuffer. 'uniform' for sampling
                random uniformly batches. 'full' for sampling the full
                DataBuffer
            data_buffer (DataBuffer): DataBuffer instance
            keys (Optional[List[str]]): DataBuffer keys for sampling
            batch_size (Optional[int], optional): batch size, needed for
                'uniform', 'backward', 'forward', defaults to None
            n_batches (Optional[int], optional): how many batches to
                sample, can be used for all modes. Note that sampling
                procedure stops in case if DataBuffer is exhausted for
                'forward' and 'backward' modes,  defaults to None
            dtype (RgArrayType, optional): dtype for sampling, can be
                either of cs.DM, np.array, torch.Tensor, defaults to
                torch.FloatTensor
            device (Optional[Union[str, torch.device]], optional):
                device to sample from, defaults to None
            fill_na (Optional[float], optional): fill value for np.nan,
                defaults to 0.0
        """
        if batch_size is None and mode in ["uniform", "backward", "forward"]:
            raise ValueError(
                "batch_size should not be None for modes ['uniform', 'backward', 'forward']"
            )
        assert mode in [
            "uniform",
            "backward",
            "forward",
            "full",
        ], "mode should be one of ['uniform', 'backward', 'forward', 'full']"
        assert not (
            n_batches is None and (mode == "uniform" or mode == "full")
        ), "'uniform' and 'full' mode are not avaliable for n_batches == None"

        BatchSampler.__init__(
            self,
            data_buffer=data_buffer,
            keys=keys,
            dtype=dtype,
            device=device,
            fill_na=fill_na,
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
        elif self.mode == "full":
            self.batch_ids = np.arange(self.len_data_buffer, dtype=int)
        else:
            raise ValueError("mode should be one of ['uniform', 'backward', 'forward']")

    def stop_iteration_criterion(self) -> bool:
        if self.mode != "full":
            if self.len_data_buffer <= self.batch_size:
                return True
        if self.mode == "forward":
            return (
                self.batch_ids[-1] >= len(self.data_buffer)
                or self.n_batches == self.n_batches_sampled
            )
        elif self.mode == "backward":
            return self.batch_ids[0] <= 0 or self.n_batches == self.n_batches_sampled
        elif self.mode == "uniform" or self.mode == "full":
            return self.n_batches == self.n_batches_sampled
        else:
            raise ValueError(
                "mode should be one of ['uniform', 'backward', 'forward', 'full']"
            )

    def next(self) -> Dict[str, RgArray]:
        batch = self.data_buffer[self.batch_ids]
        if self.mode == "forward":
            self.batch_ids += 1
        elif self.mode == "backward":
            self.batch_ids -= 1
        elif self.mode == "uniform":
            self.batch_ids = np.random.randint(
                low=0, high=self.len_data_buffer - self.batch_size
            ) + np.arange(self.batch_size, dtype=int)

        # for self.mode == "full" we should not update batch_ids as they are constant for full mode
        # i. e. self.batch_ids == np.arange(self.len_data_buffer, dtype=int)

        self.n_batches_sampled += 1
        return batch


class EpisodicSampler(BatchSampler):
    """Samples the whole episodes from DataBuffer."""

    def __init__(
        self,
        data_buffer,
        keys: List[str],
        dtype: RgArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ):
        """Instantiate a EpisodicSampler.

        Args:
            data_buffer (DataBuffer): instance of DataBuffer
            keys (List[str]): keys for sampling
            dtype (RgArrayType, optional): batch dtype for sampling, can
                be either of cs.DM, np.array, torch.Tensor, defaults to
                torch.FloatTensor
            device (Optional[Union[str, torch.device]], optional):
                torch.Tensor device for sampling, defaults to None
            fill_na (Optional[float], optional): fill value for np.nan,
                defaults to 0.0
        """
        BatchSampler.__init__(
            self,
            data_buffer=data_buffer,
            keys=keys,
            dtype=dtype,
            device=device,
            fill_na=fill_na,
        )
        self.nullify_sampler()

    def nullify_sampler(self) -> None:
        self.episode_ids = (
            self.data_buffer.to_pandas(keys=["episode_id"])
            .astype(int)
            .values.reshape(-1)
        )
        self.max_episode_id = max(self.episode_ids)
        self.cur_episode_id = min(self.episode_ids) - 1
        self.idx_batch = -1

    def stop_iteration_criterion(self) -> bool:
        return self.cur_episode_id >= self.max_episode_id

    def get_episode_batch_ids(self, episode_id) -> np.array:
        return np.arange(len(self.data_buffer), dtype=int)[
            self.episode_ids == episode_id
        ]

    def next(self) -> Dict[str, RgArray]:
        self.cur_episode_id += 1
        batch_ids = self.get_episode_batch_ids(self.cur_episode_id)
        return self.data_buffer[batch_ids]


class RandomSampler(BatchSampler):
    def __init__(
        self,
        data_buffer,
        keys: List[str],
        dtype: RgArrayType = torch.FloatTensor,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
        split_size: int = 100,
        n_splits_per_episode: int = 1,
        n_batches: int = 10,
    ):
        """Instantiate a EpisodicSampler.

        Args:
            data_buffer (DataBuffer): instance of DataBuffer
            keys (List[str]): keys for sampling
            dtype (RgArrayType, optional): batch dtype for sampling, can
                be either of cs.DM, np.array, torch.Tensor, defaults to
                torch.FloatTensor
            device (Optional[Union[str, torch.device]], optional):
                torch.Tensor device for sampling, defaults to None
            fill_na (Optional[float], optional): fill value for np.nan,
                defaults to 0.0
        """
        BatchSampler.__init__(
            self,
            data_buffer=data_buffer,
            keys=keys,
            dtype=dtype,
            device=device,
            fill_na=fill_na,
        )
        self.split_size = split_size
        self.n_splits_per_episode = n_splits_per_episode
        self.n_batches = n_batches
        self.nullify_sampler()

    def nullify_sampler(self) -> None:
        self.episode_ids = (
            self.data_buffer.to_pandas(keys=["episode_id"])
            .astype(int)
            .values.reshape(-1)
        )
        self.unique_episode_ids = np.unique(self.episode_ids)
        self.idx_batch = 0

    def stop_iteration_criterion(self) -> bool:
        return self.idx_batch >= self.n_batches

    @staticmethod
    def get_split_indexes_for_episode(
        episode_size, max_n_splits_per_episode, max_split_size, episode_start_index
    ) -> list[int]:
        n_splits_per_episode = min(max_n_splits_per_episode, episode_size)
        split_gen_range_size = episode_size // n_splits_per_episode
        split_size = min(max_split_size, split_gen_range_size)

        split_ranges = np.array(
            [
                [
                    i * split_gen_range_size,
                    (
                        (i + 1) * split_gen_range_size
                        if i < n_splits_per_episode - 1
                        else episode_size
                    )
                    - split_size
                    + 1,
                ]
                for i in range(n_splits_per_episode)
            ]
        )
        ids = []
        for sr in split_ranges:
            left = np.random.randint(low=sr[0], high=sr[1])
            right = left + split_size
            ids.extend(range(episode_start_index + left, episode_start_index + right))

        return ids

    def next(self) -> Dict[str, RgArray]:
        self.idx_batch += 1

        batch_ids = []
        for episode_id in self.unique_episode_ids:
            argwhere_episode = np.argwhere(self.episode_ids == episode_id)
            episode_start_index = argwhere_episode[0, 0]
            episode_size = len(argwhere_episode)
            batch_ids.extend(
                self.get_split_indexes_for_episode(
                    episode_size=episode_size,
                    max_n_splits_per_episode=self.n_splits_per_episode,
                    max_split_size=self.split_size,
                    episode_start_index=episode_start_index,
                )
            )

        return self.data_buffer[np.array(batch_ids)]
