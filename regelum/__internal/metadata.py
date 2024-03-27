from types import MappingProxyType

from regelum import RegelumBase


class Metadata(RegelumBase):
    def __init__(self, content):
        self.content = MappingProxyType(content)

    def __enter__(self):
        self._metadata = self.content

    def __exit__(self, exc_type, exc_val, exc_tb):
        delattr(RegelumBase, f"_{RegelumBase.__name__}__metadata")
