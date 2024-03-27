def pytest_addoption(parser):
    parser.addoption("--mode", action="store", default="basic")
