import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import logging
import numpy as np
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf, flag_override
from rcognita.visualization.vis_3wrobot import Animator3WRobot
import matplotlib.pyplot as plt


@hydra.main(
    version_base=None, config_path="../pipelines/test_conf", config_name="test",
)
def launch(cfg):
    print(cfg)


if __name__ == "__main__":
    launch()
