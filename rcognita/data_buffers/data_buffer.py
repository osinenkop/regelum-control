try:
    import torch
    from torch.utils.data import Sampler
except ImportError:
    from unittest.mock import MagicMock

    torch = MagicMock()

import numpy as np
import pandas as pd


from typing import Optional, List, Union, Any, Type, Iterable
from .types import Array, ArrayType
from .samplers import ForwardSampler
from .fifo_list import FifoList


class DataBuffer:
    def __init__(
        self,
        keys: List[str] = [
            "observation",
            "action",
            "running_objective",
            "current_total_objective",
            "episode_id",
            "timestamp",
        ],
        max_buffer_size: Optional[int] = None,
        keys_for_indexing: Optional[List[str]] = None,
        dtype_for_indexing: Optional[ArrayType] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self.keys = list(keys)
        self.initial_keys = list(keys)
        self.keys_for_indexing = (
            keys if keys_for_indexing is None else keys_for_indexing
        )
        self.dtype_for_indexing = dtype_for_indexing
        self.nullify_buffer()

    def nullify_buffer(self) -> None:
        self.data = {
            key: FifoList(max_size=self.max_buffer_size) for key in self.initial_keys
        }

    def update(self, data_in_dict_format) -> None:
        self.keys += list(data_in_dict_format.keys())
        for key, data_for_key in data_in_dict_format.items():
            self.data[key] = data_for_key

    def push_to_end(self, **kwargs) -> None:
        for key in self.keys:
            if kwargs.get(key) is not None:
                self.data[key].append(kwargs[key])

    def last(self) -> dict[str, Array]:
        return self[-1]

    def to_dict(self):
        return self.data

    def to_pandas(self, keys: Optional[List[str]] = None) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        if keys is not None:
            return df[keys]
        return df

    def __len__(self):
        return len(self.data[self.keys[0]])

    def getitem(
        self,
        idx: Union[int, slice, Any],
        keys: Optional[Union[List[str], np.array]] = None,
        dtype: Optional[ArrayType] = None,
    ) -> dict[str, Array]:
        _keys = keys if keys is not None else self.keys
        if (
            isinstance(idx, int)
            or isinstance(idx, slice)
            or isinstance(idx, np.ndarray)
        ):
            if dtype is None:
                return {key: self.data[key][idx] for key in _keys}
            elif (
                dtype == torch.tensor
                or dtype == torch.FloatTensor
                or dtype == torch.DoubleTensor
                or dtype == torch.LongTensor
            ):
                return {key: dtype(np.array(self.data[key]))[idx] for key in _keys}
            elif dtype == np.array:
                return {key: np.array(self.data[key])[idx] for key in _keys}

    def set_indexing_rules(
        self,
        keys: List[str],
        dtype: ArrayType,
    ) -> None:
        self.keys_for_indexing = keys
        self.dtype_for_indexing = dtype

    def __getitem__(self, idx) -> dict[str, Array]:
        return self.getitem(idx, self.keys_for_indexing, self.dtype_for_indexing)

    def iter_batches(
        self, sampler: Type[Sampler] = ForwardSampler, **sampler_kwargs
    ) -> Iterable[Array]:
        return sampler(data_buffer=self, **sampler_kwargs)
