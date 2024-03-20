"""Contains DataBuffer class."""

import numpy as np
import pandas as pd
import casadi as cs
import torch

from typing import Optional, List, Union, Any, Type, Iterable, Dict
from regelum.typing import RgArray, RgArrayType
from .batch_sampler import RollingBatchSampler, BatchSampler
from collections import defaultdict
from ..optimizable import OptimizerConfig


class DataBuffer:
    """DataBuffer class for storing run data.

    DataBuffer is a container for storing run data: observations, actions,
    running objectives, iteration ids, episode ids, step ids. It is designed to store any
    data of numeric format.
    """

    def __init__(
        self,
    ):
        """Instantiate a DataBuffer."""
        self.nullify_buffer()

    def delete_key(self, key) -> None:
        self.data.pop(key)

    def keys(self) -> List[str]:
        return list(self.data.keys())

    def nullify_buffer(self) -> None:
        self.data = defaultdict(list)
        self.keys_for_indexing = None
        self.dtype_for_indexing = None
        self.device_for_indexing = None
        self.fill_na_for_indexing = None

    def update(self, data_in_dict_format: dict[str, RgArray]) -> None:
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
            datum = np.array(kwargs[key])
            if np.any(np.isnan(self.data[key][-1])):
                self.data[key][-1] = datum
            else:
                self.data[key].append(datum)
                is_line_added = True

        buffer_len = len(self)
        for key in kwarg_keys.difference(current_keys):
            datum = np.array(kwargs[key])
            for _ in range(buffer_len - 1):
                self.data[key].append(np.full_like(datum, np.nan, dtype=float))
            self.data[key].append(datum)

        # if buffer len has changed fill all the rest keys with nan
        if is_line_added:
            for key in current_keys.difference(kwarg_keys):
                self.data[key].append(
                    np.full_like(self.data[key][-1], np.nan, dtype=float)
                )

    def last(self) -> dict[str, RgArray]:
        return self[-1]

    def to_dict(self):
        return self.data

    def to_pandas(
        self,
        keys: Optional[Union[List[str], Dict[str, Type[Any]]]] = None,
    ) -> pd.DataFrame:
        if isinstance(keys, list):
            _keys = {k: float for k in keys}
        elif isinstance(keys, dict):
            assert set(keys.keys()).issubset(
                self.data.keys()
            ), "keys must be a subset of data keys"
            _keys = keys
        elif keys is None:
            _keys = {k: float for k in self.data.keys()}
        else:
            raise AssertionError("keys must be a list or a dict or None")

        return pd.DataFrame(
            {
                k: (
                    np.array(self.data[k], dtype=_keys[k]).reshape(-1)
                    if self.data[k][0].size == 1
                    else self.data[k]
                )
                for k in _keys
            }
        )

    def __len__(self):
        if len(self.data.keys()) == 0:
            return 0
        else:
            return max([len(self.data[k]) for k in self.data.keys()])

    def _fill_na(self, arr: np.array, fill_na: Optional[float] = None) -> np.array:
        if fill_na is None:
            return arr
        else:
            np.nan_to_num(arr, copy=False, nan=fill_na)
            return arr

    def getitem(
        self,
        idx: Union[int, slice, Any],
        keys: Optional[Union[List[str], np.array]] = None,
        dtype: RgArrayType = np.array,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ) -> dict[str, RgArray]:
        _keys = keys if keys is not None else self.data.keys()
        difference_keys_set = set(_keys).difference(self.data.keys())
        if len(difference_keys_set) > 0:
            raise ValueError(
                f"Unexpected keys in DataBuffer.getitem: {difference_keys_set}"
            )

        if (
            isinstance(idx, int)
            or isinstance(idx, slice)
            or isinstance(idx, np.ndarray)
        ):
            if dtype == np.array:
                return {
                    key: self._fill_na(np.vstack(self.data[key])[idx], fill_na=fill_na)
                    for key in _keys
                }
            elif (
                dtype == torch.tensor
                or dtype == torch.FloatTensor
                or dtype == torch.DoubleTensor
                or dtype == torch.LongTensor
            ):
                if device is not None:
                    return {
                        key: dtype(
                            self._fill_na(np.vstack(self.data[key]), fill_na=fill_na)
                        )[idx].to(device)
                        for key in _keys
                    }
                else:
                    return {
                        key: dtype(
                            self._fill_na(np.vstack(self.data[key]), fill_na=fill_na)
                        )[idx]
                        for key in _keys
                    }
            elif dtype == cs.DM:
                return {
                    key: dtype(
                        self._fill_na(np.vstack(self.data[key]), fill_na=fill_na)[idx]
                    )
                    for key in _keys
                }
            else:
                raise ValueError(f"Unexpeted dtype in data_buffer.getitem: {dtype}")

    def set_indexing_rules(
        self,
        keys: List[str],
        dtype: RgArrayType,
        device: Optional[Union[str, torch.device]] = None,
        fill_na: Optional[float] = 0.0,
    ) -> None:
        self.keys_for_indexing = keys
        self.dtype_for_indexing = dtype
        self.device_for_indexing = device
        self.fill_na_for_indexing = fill_na

    def __getitem__(self, idx) -> dict[str, RgArray]:
        return self.getitem(
            idx,
            keys=self.keys_for_indexing,
            dtype=self.dtype_for_indexing,
            device=self.device_for_indexing,
            fill_na=self.fill_na_for_indexing,
        )

    def concat(self, other):
        for key in other.keys():
            self.data[key] += other.data[key]

    def get_latest(self, key: str) -> np.array:
        return self.data[key][-1]

    def iter_batches(
        self,
        keys: List[str],
        batch_sampler: Type[BatchSampler] = RollingBatchSampler,
        **batch_sampler_kwargs,
    ) -> Iterable[RgArray]:
        return batch_sampler(data_buffer=self, keys=keys, **batch_sampler_kwargs)

    def sample_last(
        self, keys: List[str], dtype: RgArrayType, n_samples: int = 1
    ) -> dict[str, RgArray]:
        if n_samples <= len(self):
            return self.getitem(
                np.arange(-n_samples, 0, dtype=int), keys=keys, dtype=dtype
            )

    def get_optimization_kwargs(
        self, keys: List[str], optimizer_config: OptimizerConfig
    ):
        config_options = optimizer_config.config_options

        method_name = config_options.get("data_buffer_sampling_method")
        kwargs = config_options.get("data_buffer_sampling_kwargs")
        assert (
            method_name is not None
        ), "Specify `data_buffer_sampling_method` in your optimizer_config"
        assert (
            kwargs is not None
        ), "Specify `data_buffer_sampling_kwargs` in your optimizer_config"

        if method_name == "iter_batches":
            return {"batch_sampler": self.iter_batches(keys=keys, **kwargs)}
        elif method_name == "sample_last":
            return self.sample_last(keys=keys, **kwargs)
        else:
            raise ValueError("Unknown data_buffer_sampling_method")
