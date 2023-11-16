import sys, os
import numpy as np

# ["2tank", "3wrobot", "3wrobot_ni", "cartpole", "inv_pendulum", "kin_point", "lunar_lander"])
# ["ddpg", "ddqn", "dqn", "dqn", "mpc", "pg", "pid", "rpo", "rpo_deep", "rql", "sarsa", "sdpg", "sql"]


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
        sys.argv.insert(1, "pipeline.sampling_time=0.1")
        # sys.argv.insert(1, "pipeline.N_episodes=2")
        # sys.argv.insert(1, "pipeline.N_iterations=2")
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
        return f"{self.params['system']}_{self.params['pipeline']}"


class MPCTest(TestSetup):
    def __init__(self, system):
        super().__init__(
            system=system,
            pipeline="mpc",
            **{"pipeline.actor.predictor.prediction_horizon": 2},
        )


#######################################################################################################################
#######################################################################################################################
pipelines = (
    "sdpg",
    "ppo",
    "ddpg",
    "reinforce",
    # "dqn",
    # "sarsa",
    # "rpo",
    # "rql",
    # "rql_torch",
    # "sql",
    # "sql_torch",
    # "rpo_torch",
    # "mpc_torch",
    "mpc",
    "mpc",
)
systems = "3wrobot_ni", "cartpole", "inv_pendulum", "kin_point", "2tank", "lunar_lander"


basic = [
    TestSetup(
        system=system,
        pipeline=pipeline,
        **(
            {"simulator.time_final": 3.0}
            | (
                {"pipeline.N_iterations": 2, "pipeline.N_episodes": 2}
                if pipeline != "mpc"
                else {}
            )
        ),
    )
    for pipeline, system in zip(pipelines, np.tile(systems, len(pipelines)))
]

# basic += [
#     TestSetup(
#         system="3wrobot_ni",
#         pipeline=pipeline,
#         **{
#             "simulator.time_final": 3.0,
#             "constraint_parser": "constant_parser",
#             "prefix": "constraint",
#         },
#     )
#     for pipeline in ["rpo", "sql", "mpc", "rql"]
# ]

# basic += [
#     TestSetup(
#         system="3wrobot_ni",
#         pipeline=pipeline,
#         **{
#             "simulator.time_final": 0.5,
#             "constraint_parser": "constant_parser",
#             "prefix": "constraint_torch",
#             "pipeline.policy.prediction_horizon": 1,
#             "pipeline/policy/optimizer_config": "online_torch_sgd",
#             "pipeline.policy.optimizer_config.config_options.n_epochs": 1,
#             "pipeline.policy.optimizer_config.config_options.constrained_optimization_policy.defaults.n_epochs_per_constraint": 1,
#         },
#     )
#     for pipeline in ["mpc_torch"]
# ]

# basic += [
#     TestSetup(
#         system="3wrobot_ni",
#         pipeline=pipeline,
#         **{
#             "simulator.time_final": 3.0,
#             "prefix": "truncated_noise",
#             "pipeline/policy/model": "perceptron_with_truncated_normal_noise",
#             "pipeline.sampling_time": 0.1,
#         },
#     )
#     for pipeline in ["ppo", "reinforce", "sdpg"]
# ]


full = sum(
    [
        [
            TestSetup(system=system, pipeline="sdpg", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, pipeline="ppo", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, pipeline="ddpg"),
            TestSetup(system=system, pipeline="reinforce"),
            TestSetup(system=system, pipeline="dqn", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, pipeline="sarsa", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, pipeline="rpo", **{"simulator.time_final": 3.0}),
            TestSetup(system=system, pipeline="rql", **{"simulator.time_final": 3.0}),
            TestSetup(
                system=system, pipeline="rql_torch", **{"simulator.time_final": 3.0}
            ),
            TestSetup(system=system, pipeline="sql", **{"simulator.time_final": 3.0}),
            TestSetup(
                system=system, pipeline="sql_torch", **{"simulator.time_final": 3.0}
            ),
            TestSetup(
                system=system,
                pipeline="rpo_torch",
                **{"simulator.time_final": 3.0},
            ),
            TestSetup(
                system=system,
                pipeline="mpc_torch",
                **{"simulator.time_final": 3.0},
            ),
            TestSetup(system=system, pipeline="mpc", **{"simulator.time_final": 3.0}),
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
         TestSetup(system="3wrobot", pipeline="rpo"),
         TestSetup(system="3wrobot_ni", pipeline="rpo"),
         TestSetup(system="cartpole", pipeline="rql"),
         TestSetup(system="inv_pendulum", pipeline="rpo"),
         MPCTest(system="kin_point"),
         MPCTest(system="lunar_lander")]

extended = basic + \
           [TestSetup(system="kin_point", pipeline="ddpg"),
            TestSetup(system="kin_point", pipeline="sarsa")]
"""
