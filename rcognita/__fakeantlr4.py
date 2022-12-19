from antlr4.Token import Token
'''
When we have the time, let's instead patch 
hydra._internal.core_plugins.file_config_source.FileConfigSource.load_config
We could insert a fake file descriptor that reads stuff from a string (that we already read and edited).
We could even just fork hydra, although I'm worried that this may hinder compatibility with new hydra plugins in the future.
But then again, reforking it shouldn't be too hard. Perhaps this is what we should do. This will give us even greatre flexibility.
'''

class InputStream (object):
    #__slots__ = ('name', 'strdata', '_index', 'data', '_size')

    def __init__(self, data: str):
        self.name = "<empty>"
        self.strdata = data.replace('(', '\\(').replace(')', '\\)').replace('[', '\\[').replace(']', '\\]').replace(',', '\\,').replace(';', ',')
        self._loadString()

    def _loadString(self):
        self._index = 0
        self.data = [ord(c) for c in self.strdata]
        self._size = len(self.data)

    @property
    def index(self):
        return self._index

    @property
    def size(self):
        return self._size

    # Reset the stream so that it's in the same state it was
    #  when the object was created *except* the data array is not
    #  touched.
    #
    def reset(self):
        self._index = 0

    def consume(self):
        if self._index >= self._size:
            assert self.LA(1) == Token.EOF
            raise Exception("cannot consume EOF")
        self._index += 1

    def LA(self, offset: int):
        if offset==0:
            return 0 # undefined
        if offset<0:
            offset += 1 # e.g., translate LA(-1) to use offset=0
        pos = self._index + offset - 1
        if pos < 0 or pos >= self._size: # invalid
            return Token.EOF
        return self.data[pos]

    def LT(self, offset: int):
        return self.LA(offset)

    # mark/release do nothing; we have entire buffer
    def mark(self):
        return -1

    def release(self, marker: int):
        pass

    # consume() ahead until p==_index; can't just set p=_index as we must
    # update line and column. If we seek backwards, just set p
    #
    def seek(self, _index: int):
        if _index<=self._index:
            self._index = _index # just jump; don't update stream state (line, ...)
            return
        # seek forward
        self._index = min(_index, self._size)

    def getText(self, start :int, stop: int):
        if stop >= self._size:
            stop = self._size-1
        if start >= self._size:
            return ""
        else:
            return self.strdata[start:stop+1]

    def __str__(self):
        return self.strdata
