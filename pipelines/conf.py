import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)


import rcognita as r


@r.main(
    version_base=None, config_path="../pipelines/conf/", config_name="3wrobot",
)
def launch(cfg):
    actor = ~cfg.actor
    print(actor.critic, actor.running_objective, actor.optimizer)


if __name__ == "__main__":
    launch()
