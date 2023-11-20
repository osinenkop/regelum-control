import sys, os
import numpy as np

# ["2tank", "3wrobot", "3wrobot_ni", "cartpole", "inv_pendulum", "kin_point", "lunar_lander"])
# ["ddpg", "ddqn", "dqn", "dqn", "mpc", "pg", "pid", "rpv", "rpv_deep", "rql", "sarsa", "sdpg", "sql"]


class TestSetup:
    def __init__(self, config_path=None, config_name=None, **params):
        self.config_path = (
            config_path if config_path is not None else self.config_path_default
        )
        self.config_path = (
            os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
            + f"/../../presets/{self.config_path}"
        )
        self.config_name = (
            config_name if config_name is not None else self.config_name_default
        )
        self.params = params

    @property
    def config_path_default(self):
        return "stable-presets"
        # return os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../presets/general")

    @property
    def config_name_default(self):
        return "main"

    def _pre_setup(self):
        os.environ["REHYDRA_FULL_ERROR"] = "1"
        sys.argv.insert(1, "--single-thread")
        sys.argv.insert(1, "--no-git")
        sys.argv.insert(1, "disallow_uncommitted=False")
        sys.argv.insert(1, "simulator.time_final=1")
        sys.argv.insert(1, "scenario.sampling_time=0.1")
        # sys.argv.insert(1, "scenario.N_episodes=2")
        # sys.argv.insert(1, "scenario.N_iterations=2")
        sys.argv.insert(1, "--experiment=TESTS")

    def __call__(self):
        sys.argv = [sys.argv[0]]
        for param in self.params:
            sys.argv.insert(1, f"{param}={self.params[param]}")
        self._pre_setup()
        os.environ[
            "REGELUM_RECENT_TEST_INFO"
        ] = f"Command line arguments: {' '.join(sys.argv[1:])}\nConfig: {os.path.abspath(self.config_path)}/{self.config_name}.yaml"
        return {"config_path": self.config_path, "config_name": self.config_name}

    def __str__(self):
        return f"{self.params['system']}_{self.params['scenario']}"


class MPCTest(TestSetup):
    def __init__(self, system):
        super().__init__(
            system=system,
            scenario="mpc",
            **{"scenario.actor.predictor.prediction_horizon": 2},
        )


#######################################################################################################################
#######################################################################################################################
scenarios = (
    "sdpg",
    "ppo",
    "ddpg",
    "reinforce",
    # "dqn",
    # "sarsa",
    "rpv",
    "rql",
    # "rql_torch",
    "sql",
    # "sql_torch",
    # "rpv_torch",
    # "mpc_torch",
    "mpc",
)
systems = "3wrobot_ni", "cartpole", "inv_pendulum", "kin_point", "2tank", "lunar_lander"


def get_kwargs(scenario):
    if scenario == "mpc":
        return {}
    elif scenario in ["reinforce", "ppo", "sdpg", "ddpg"]:
        return {"scenario.N_iterations": 2, "scenario.N_episodes": 2}
    else:
        return {"scenario.N_iterations": 2}


basic = [
    TestSetup(
        system=system,
        scenario=scenario,
        **({"simulator.time_final": 3.0} | get_kwargs(scenario)),
    )
    for scenario, system in zip(scenarios, np.tile(systems, len(scenarios)))
]

basic += [
    TestSetup(
        system="3wrobot_ni",
        scenario=scenario,
        **{
            "simulator.time_final": 3.0,
            "constraint_parser": "constant_parser",
            "prefix": "constraint",
        },
    )
    for scenario in ["rpv", "sql", "mpc", "rql"]
]

basic += [
    TestSetup(
        system="3wrobot_ni",
        scenario="calf",
    )
]

basic += [
    TestSetup(
        system="3wrobot_ni",
        scenario="nominal",
    )
]

# basic += [
#     TestSetup(
#         system="3wrobot_ni",
#         scenario=scenario,
#         **{
#             "simulator.time_final": 0.5,
#             "constraint_parser": "constant_parser",
#             "prefix": "constraint_torch",
#             "scenario.policy.prediction_horizon": 1,
#             "scenario/policy/optimizer_config": "online_torch_sgd",
#             "scenario.policy.optimizer_config.config_options.n_epochs": 1,
#             "scenario.policy.optimizer_config.config_options.constrained_optimization_policy.defaults.n_epochs_per_constraint": 1,
#         },
#     )
#     for scenario in ["mpc_torch"]
# ]

# basic += [
#     TestSetup(
#         system="3wrobot_ni",
#         scenario=scenario,
#         **{
#             "simulator.time_final": 3.0,
#             "prefix": "truncated_noise",
#             "scenario/policy/model": "perceptron_with_truncated_normal_noise",
#             "scenario.sampling_time": 0.1,
#         },
#     )
#     for scenario in ["ppo", "reinforce", "sdpg"]
# ]


full = sum(
    [
        [
            TestSetup(system=system, scenario="sdpg", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, scenario="ppo", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, scenario="ddpg"),
            TestSetup(system=system, scenario="reinforce"),
            TestSetup(system=system, scenario="dqn", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, scenario="sarsa", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, scenario="rpv", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, scenario="rql", **{"simulator.time_final": 3.0}),
            TestSetup(
                system=system, scenario="rql_torch", **{"simulator.time_final": 3.0}
            ),
            TestSetup(system=system, scenario="sql", **{"simulator.time_final": 3.0}),
            TestSetup(
                system=system, scenario="sql_torch", **{"simulator.time_final": 3.0}
            ),
            TestSetup(
                system=system,
                scenario="rpv_torch",
                **{"simulator.time_final": 3.0},
            ),
            TestSetup(
                system=system,
                scenario="mpc_torch",
                **{"simulator.time_final": 3.0},
            ),
            TestSetup(system=system, scenario="mpc", **{"simulator.time_final": 3.0}),
        ]
        for system in [
            # "2tank",
            # "3wrobot",
            "3wrobot_ni",
            "cartpole",
            "inv_pendulum",
            "kin_point",
            # "lunar_lander",
        ]
    ],
    [],
)
"""
basic = [MPCTest(system="2tank"),
         TestSetup(system="3wrobot", scenario="rpv"),
         TestSetup(system="3wrobot_ni", scenario="rpv"),
         TestSetup(system="cartpole", scenario="rql"),
         TestSetup(system="inv_pendulum", scenario="rpv"),
         MPCTest(system="kin_point"),
         MPCTest(system="lunar_lander")]

extended = basic + \
           [TestSetup(system="kin_point", scenario="ddpg"),
            TestSetup(system="kin_point", scenario="sarsa")]
"""
