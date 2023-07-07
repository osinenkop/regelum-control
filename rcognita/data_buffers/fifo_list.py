from typing import Any, Optional


class FifoList(list):
    def __init__(self, *args, max_size: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def append(self, item: Any) -> None:
        super().append(item)
        if self.max_size is not None:
            if len(self) > self.max_size:
                self.pop(0)
