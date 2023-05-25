import pytest
import sys, os


import rcognita as rc

@pytest.fixture(params=["2tank", "3wrobot", "3wrobot_ni", "cartpole", "inv_pendulum", "kin_point", "lunar_lander"])
def system(request):
    return request.param

@pytest.fixture(params=["ddpg", "ddqn", "dqn", "dqn", "mpc", "pg", "pid", "rpo", "rpo_deep", "rql", "sarsa", "sdpg", "sql"])
def controller(request):
    return request.param

@pytest.fixture
def playground_dir():
    return os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../playground")

@pytest.fixture
def launch(system, playground_dir):
    sys.argv = [sys.argv[0]]
    sys.argv.insert(1, f"system={system}")
    sys.argv.insert(1, f"controller=mpc")
    sys.argv.insert(1, f"controller.actor.predictor.prediction_horizon=2")
    sys.argv.insert(1, "disallow_uncommitted=False")
    sys.argv.insert(1, "simulator.time_final=1")
    sys.argv.insert(1, "controller.sampling_time=0.5")
    sys.argv.insert(1, "scenario.N_episodes=2")
    sys.argv.insert(1, "--single-thread")

    @rc.main(config_path=playground_dir + "/general", config_name="main")
    def launch(cfg):

        scenario = ~cfg.scenario
        if scenario.howanim in rc.ANIMATION_TYPES_REQUIRING_ANIMATOR:
            try:
                animator = ~cfg.animator
            except:
                raise NotImplementedError("Can't instantiate animator for your system")
            if scenario.howanim == "live":
                animator.play_live()
            elif scenario.howanim in rc.ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK:
                scenario.run()
                animator.playback()
        else:
            scenario.run()

        return 0

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



