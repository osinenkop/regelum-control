import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import numpy
import rcognita as r
import omegaconf
from omegaconf import DictConfig, OmegaConf, flag_override
import hydra
from hydra.utils import instantiate


@r.main(
    version_base=None, config_path="../rcognita/conf", config_name="config_test",
)
def my_app(cfg: DictConfig):
    print(type(cfg.simulation.state_init))


if __name__ == "__main__":
    my_app()
