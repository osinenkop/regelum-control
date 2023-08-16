"""Constains FifoList class."""

from typing import Any, Optional


class FifoList(list):
    """Fifo list class. Acts like list but when size exceeds max_size it removes the first element."""

    def __init__(self, *args, max_size: Optional[int] = None, **kwargs):
        """Instantiate a FifoList.

        :param max_size: maximum size of the list. If `None` the list is not limited in size, defaults to None
        :type max_size: Optional[int], optional
        """
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def append(self, item: Any) -> None:
        super().append(item)
        if self.max_size is not None:
            if len(self) > self.max_size:
                self.pop(0)
