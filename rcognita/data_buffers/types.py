try:
    import torch
except ImportError:
    from unittest.mock import MagicMock

    torch = MagicMock()

import numpy as np
from .fifo_list import FifoList
from typing import Union, Type


ArrayType = Union[
    Type[np.array],
    Type[torch.tensor],
    Type[torch.FloatTensor],
    Type[torch.DoubleTensor],
    Type[torch.LongTensor],
]

Array = Union[
    FifoList,
    np.array,
    torch.tensor,
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
]
