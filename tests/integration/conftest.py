import pytest
from .setup import *
import regelum as rc


def pytest_generate_tests(metafunc):
    mode = metafunc.config.getoption("mode")
    try:
        setups = eval(mode)
        ids = list(map(str, setups))
        metafunc.parametrize("setup", setups, indirect=True, ids=ids)
    except NameError:
        raise ValueError(
            f'Invalid testing mode "{mode}". See tests/integration/setup.py for declared testing modes.'
        )


@pytest.fixture
def setup(request):
    return request.param


@pytest.fixture
def launch(setup):
    @rc.main(**setup())
    def launch(cfg):
        controller = ~cfg.controller
        controller.run()

    return launch


@pytest.fixture
def output(launch):
    return launch()


@pytest.fixture
def result(output):
    return output["result"][0]


@pytest.fixture
def config(output):
    return output["cfg"][0]


@pytest.fixture
def overrides(output):
    return output["overrides"][0]


@pytest.fixture
def directory(output):
    return output["directory"][0]
