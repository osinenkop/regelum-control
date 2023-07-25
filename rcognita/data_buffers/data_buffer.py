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
from collections import defaultdict


class DataBuffer:
    def __init__(
        self,
        max_buffer_size: Optional[int] = None,
        keys_for_indexing: Optional[List[str]] = None,
        dtype_for_indexing: Optional[ArrayType] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self.keys_for_indexing = keys_for_indexing
        self.dtype_for_indexing = dtype_for_indexing
        self.nullify_buffer()

    def nullify_buffer(self) -> None:
        self.data = defaultdict(lambda: FifoList(max_size=self.max_buffer_size))

    def update(self, data_in_dict_format: dict[str, Array]) -> None:
        for key, data_for_key in data_in_dict_format.items():
            self.data[key] = data_for_key

    def push_to_end(self, **kwargs) -> None:
        for key, data_item_for_key in kwargs.items():
            self.data[key].append(data_item_for_key)

    def last(self) -> dict[str, Array]:
        return self[-1]

    def to_dict(self):
        return self.data

    def to_pandas(self, keys: Optional[List[str]] = None) -> pd.DataFrame:
        if keys is not None:
            return pd.DataFrame({k: self.data[k] for k in keys})

        return pd.DataFrame(self.data)

    def __len__(self):
        return max([len(self.data[k]) for k in self.data.keys()])

    def getitem(
        self,
        idx: Union[int, slice, Any],
        keys: Optional[Union[List[str], np.array]] = None,
        dtype: Optional[ArrayType] = None,
    ) -> dict[str, Array]:
        _keys = keys if keys is not None else self.data.keys
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
