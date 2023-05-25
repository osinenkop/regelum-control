

def test_no_error(result):
    if isinstance(result, Exception):
        raise result
