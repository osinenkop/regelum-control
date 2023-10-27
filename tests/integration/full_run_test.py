import os


class FailedTest(Exception):
    pass


def test_no_error(result):
    if isinstance(result, Exception):
        raise FailedTest(
            f"Testee raised an unhandled {type(result).__name__} (see chained exception).\n {os.environ['REGELUM_RECENT_TEST_INFO']}"
        ) from result
