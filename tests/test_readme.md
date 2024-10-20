# A brief guide to testing `regelum`

Before you run the tests it is advised that you run `scripts/developer-setup.sh` first.
This way necessary dependencies will be installed and tests will be run automatically, when attempting to commit.
Alternatively, you could manualy install the packages listed in `requirements-dev.txt`.

To run the default set of test you only need to run the following line from repo root:
```angular2html
pytest
```

If you instead want to run tests in parallel, simply use `-n` to specify the number of processes to use.
For instance if you want to run tests with 10 parallel processes, you would need to run:
```angular2html
pytest -n 10
```

Also don't forget to lint your code with:
```angular2html
ruff ./rcognita
```

## Creating integration tests


The only module you really need to edit is `tests.integration.setup`.

At the end of the module you'll see lists of the following kind:

```angular2html
basic = [MPCTest(system="2tank"),
         TestSetup(system="3wrobot_dyn", pipeline="rpo"),
         TestSetup(system="3wrobot_kin", pipeline="rpo"),
         TestSetup(system="cartpole", pipeline="rql"),
         TestSetup(system="pendulum", pipeline="rpo"),
         MPCTest(system="kin_point"),
         MPCTest(system="lunar_lander")]

extended = basic + \
           [TestSetup(system="kin_point", pipeline="ddpg"),
            TestSetup(system="kin_point", pipeline="sarsa")]
```

Each of those lists corresponds to a set of tests. Each entry is an individual test.
So for instance, if you were to execute
```
pytest --mode=extended
```
it would run nine tests that the list `extended` contains. 
You can add your own lists of this kind and their names will be recognized when setting `--mode`.
The default value for `--mode` is `basic`.

The keyword arguments for `TestSetup` are `config_path` and `config_name` and they correspond to respective paths and
names as if your current dir was `presets`. Other keyword arguments like `system` and `pipeline` will create overrides with respective values.
For instance `TestSetup(simulator="x")` would change the simulator configuration to `x.yaml`.

All other classes that you see instantiated inside these lists (like `MPCTest`) are merely derived from `TestSetup` for convenience.

# FAQ

## Why did the tests start failing after I renamed my project?

Delete `__pycache__` and `.pytest_cache` from everywhere, including `.`, `./tests`, `./tests/integration`, `./regelum`.


