class A:
    def __init__(self, a=1):
        self.a = a


class B(A):
    def __init__(self, b=True, **kwargs):
        super().__init__(**kwargs)
        self.b = b


class C(A):
    def __init__(self, b=False, **kwargs):
        super().__init__(**kwargs)
        self.b = b


class D(A):
    def __init__(self, d="kek", **kwargs):
        super().__init__(**kwargs)
        self.d = d
