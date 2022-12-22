import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)


import rcognita as r


@r.main(
    version_base=None,
    config_path="../pipelines/conf/",
    config_name="episodic_scenario",
)
def launch(scenario):
    scenario = ~scenario
    scenario.run()


if __name__ == "__main__":
    launch()
