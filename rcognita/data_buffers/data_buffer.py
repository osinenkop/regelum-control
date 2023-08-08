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
from .samplers import RollingSampler
from .fifo_list import FifoList
from collections import defaultdict


class DataBuffer:
    def __init__(
        self,
        max_buffer_size: Optional[int] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self.nullify_buffer()

    def delete_key(self, key) -> None:
        self.data.pop(key)

    def keys(self) -> List[str]:
        return list(self.data.keys())

    def nullify_buffer(self) -> None:
        self.data = defaultdict(lambda: FifoList(max_size=self.max_buffer_size))
        self.keys_for_indexing = None
        self.dtype_for_indexing = None

    def update(self, data_in_dict_format: dict[str, Array]) -> None:
        for key, data_for_key in data_in_dict_format.items():
            self.data[key] = data_for_key

    def push_to_end(self, **kwargs) -> None:
        current_keys = set(self.data.keys())
        kwarg_keys = set(kwargs.keys())

        for _, data_item_for_key in kwargs.items():
            if np.any(np.isnan(data_item_for_key)):
                raise ValueError(
                    f"{type(data_item_for_key)} nan values are not allowed for `push_to_end` in data buffer"
                )
        is_line_added = False
        for key in current_keys.intersection(kwarg_keys):
            if np.any(np.isnan(self.data[key][-1])):
                self.data[key][-1] = kwargs[key]
            else:
                self.data[key].append(kwargs[key])
                is_line_added = True

        buffer_len = len(self)
        for key in kwarg_keys.difference(current_keys):
            for _ in range(buffer_len - 1):
                self.data[key].append(np.full_like(kwargs[key], np.nan, dtype=float))
            self.data[key].append(kwargs[key])

        # if buffer len has changed fill all the rest keys with nan
        if is_line_added:
            for key in current_keys.difference(kwarg_keys):
                self.data[key].append(
                    np.full_like(self.data[key][-1], np.nan, dtype=float)
                )

    def last(self) -> dict[str, Array]:
        return self[-1]

    def to_dict(self):
        return self.data

    def to_pandas(self, keys: Optional[List[str]] = None) -> pd.DataFrame:
        if keys is not None:
            return pd.DataFrame({k: self.data[k] for k in keys})

        return pd.DataFrame(self.data)

    def __len__(self):
        if len(self.data.keys()) == 0:
            return 0
        else:
            return max([len(self.data[k]) for k in self.data.keys()])

    def getitem(
        self,
        idx: Union[int, slice, Any],
        keys: Optional[Union[List[str], np.array]] = None,
        dtype: Optional[ArrayType] = None,
    ) -> dict[str, Array]:
        _keys = keys if keys is not None else self.data.keys()
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
        self, sampler: Type[Sampler] = RollingSampler, **sampler_kwargs
    ) -> Iterable[Array]:
        return sampler(data_buffer=self, **sampler_kwargs)
