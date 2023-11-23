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
        sys.argv.insert(1, "simulator.time_final=1.3")
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

systems = "3wrobot_ni", "cartpole", "inv_pendulum", "kin_point", "2tank", "lunar_lander"

scenarios_overrides = {
    "sdpg": {
        "scenario.critic_n_epochs": 1,
        "scenario.policy_n_epochs": 1,
        "scenario.N_episodes": 2,
        "scenario.N_iterations": 2,
        "scenario.critic_td_n": 2,
    },
    "ppo": {
        "scenario.critic_n_epochs": 1,
        "scenario.policy_n_epochs": 1,
        "scenario.N_episodes": 1,
        "scenario.N_iterations": 1,
    },
    "ddpg": {
        "scenario.critic_n_epochs": 1,
        "scenario.policy_n_epochs": 1,
        "scenario.N_episodes": 1,
        "scenario.N_iterations": 1,
    },
    "reinforce": {
        "scenario.policy_n_epochs": 1,
        "scenario.N_episodes": 1,
        "scenario.N_iterations": 1,
    },
    "sql": {
        "scenario.prediction_horizon": 2,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
    },
    "rql": {
        "scenario.prediction_horizon": 2,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
    },
    "rpv": {
        "scenario.prediction_horizon": 2,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
    },
    "mpc": {"scenario.prediction_horizon": 2},
    "mpc_torch": {"scenario.n_epochs": 1, "scenario.prediction_horizon": 2},
    "calf": {
        "scenario.prediction_horizon": 1,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
    },
    "calf_torch": {
        "scenario.prediction_horizon": 1,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
        "scenario.policy_n_epochs": 1,
        "scenario.critic_n_epochs": 1,
        "scenario.critic_n_epochs_per_constraint": 1,
    },
    "sql_torch": {
        "scenario.prediction_horizon": 2,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
        "scenario.policy_n_epochs": 1,
    },
    "rpv_torch": {
        "scenario.prediction_horizon": 2,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
        "scenario.policy_n_epochs": 1,
    },
    "sql_torch": {
        "scenario.prediction_horizon": 2,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
        "scenario.policy_n_epochs": 1,
    },
    "dqn": {
        "scenario.N_iterations": 1,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
        "scenario.policy_n_epochs": 1,
        "scenario.critic_n_epochs": 1,
        "scenario.size_mesh": 5,
    },
    "sarsa": {
        "scenario.N_iterations": 1,
        "scenario.N_iterations": 1,
        "scenario.critic_batch_size": 2,
        "scenario.critic_td_n": 1,
        "scenario.policy_n_epochs": 1,
        "scenario.critic_n_epochs": 1,
    },
}


basic = [
    TestSetup(system=system, scenario=scenario, **scenarios_overrides[scenario])
    for scenario, system in zip(
        scenarios_overrides.keys(), np.tile(systems, len(scenarios_overrides))
    )
]

basic += [
    TestSetup(
        system="3wrobot_ni",
        scenario=scenario,
        **(
            {
                "constraint_parser": "constant_parser",
                "prefix": "constraint",
            }
            | scenarios_overrides[scenario]
        ),
    )
    for scenario in ["rpv", "sql", "mpc", "rql"]
]

basic += [TestSetup(system=system, scenario="nominal") for system in systems]
