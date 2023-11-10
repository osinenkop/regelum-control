from regelum.data_buffers import DataBuffer
import numpy as np
import pytest


@pytest.fixture()
def trivial_data_buffer():
    data_buffer = DataBuffer()

    for i in range(10):
        data_buffer.push_to_end(key1=i, key2=[i, i], key3=[[i, i, i]])

    return data_buffer


def test_length(trivial_data_buffer):
    assert len(trivial_data_buffer) == 10, "Unexpected length of DataBuffer"


def test_getitem_int_idx(trivial_data_buffer: DataBuffer):
    first_value, middle_value, last_value = (
        trivial_data_buffer.getitem(idx=0),
        trivial_data_buffer.getitem(idx=5),
        trivial_data_buffer.getitem(idx=-1),
    )
    assert (
        isinstance(first_value, dict)
        and isinstance(middle_value, dict)
        and isinstance(last_value, dict)
    )
    assert np.allclose(first_value["key1"], np.array([0]))
    assert np.allclose(last_value["key2"], np.array([9, 9]))
    assert np.allclose(middle_value["key3"], np.array([5, 5, 5]))


def test_slicing(trivial_data_buffer: DataBuffer):
    values = trivial_data_buffer.getitem(idx=slice(3, 5))

    assert np.allclose(values["key1"], np.array([3, 4]).reshape(-1, 1))
    assert np.allclose(values["key2"], np.array([[3, 3], [4, 4]]))
    assert np.allclose(values["key3"], np.array([[3, 3, 3], [4, 4, 4]]))


def test_getitem_ids(trivial_data_buffer: DataBuffer):
    values = trivial_data_buffer.getitem(idx=np.array([3, 4]))

    assert np.allclose(values["key1"], np.array([3, 4]).reshape(-1, 1))
    assert np.allclose(values["key2"], np.array([[3, 3], [4, 4]]))
    assert np.allclose(values["key3"], np.array([[3, 3, 3], [4, 4, 4]]))


# @pytest.raises(ValueError)
# def test_raise_unexpected_key(trivial_data_buffer: DataBuffer):
#     trivial_data_buffer.getitem(idx=1, keys=["key4"])
