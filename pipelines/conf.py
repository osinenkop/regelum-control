import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import logging
import numpy as np
import rcognita as r
import omegaconf
from omegaconf import DictConfig, OmegaConf, flag_override



@r.main(
    config_path="../pipelines/conf/",
    config_name="episodic_scenario",
)
def my_app(cfg: r.ComplementedConfig):
    print(~cfg.actor, cfg.is_log)


if __name__ == "__main__":
    my_app()
