"""Contains type prototyping for DataBuffers."""


try:
    import torch
except ImportError:
    from unittest.mock import MagicMock

    torch = MagicMock()

import numpy as np
import casadi as cs
from .fifo_list import FifoList
from typing import Union, Type


RgArrayType = Union[
    Type[np.array],
    Type[torch.Tensor],
    Type[torch.FloatTensor],
    Type[torch.DoubleTensor],
    Type[torch.LongTensor],
    Type[cs.DM],
    Type[cs.MX],
]

RgArray = Union[
    FifoList,
    np.array,
    torch.Tensor,
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
    cs.DM,
    cs.MX,
]
