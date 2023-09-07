Initializing a new batch sampler
==================



If you want to initialize a new batch sampler then create the ancestor of base class `BatchSampler`
::
    from rcognita.data_buffer import BatchSampler

    class MyBatchSampler(BatchSampler):
        def __init__(
            self,
            data_buffer,
            keys,
        ):
            super().__init__()

and redefine 


