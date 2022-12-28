class A:
    def __init__(self, a=1):
        self.a = a


class B(A):
    def __init__(self, b=True, **kwargs):
        super().__init__(**kwargs)
        self.b = b


class C(A):
    def __init__(self, b=False, foo=None, **kwargs):
        super().__init__(**kwargs)
        self.foo = foo
        self.b = b


class D(A):
    def __init__(self, d="kek", **kwargs):
        super().__init__(**kwargs)
        self.d = d


class Foo:
    def __init__(self, var=0):
        self.var = var
