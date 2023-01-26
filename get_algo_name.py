from omegaconf import OmegaConf

path = "/mnt/abolychev/rcognita-gitlab/rcognita/presets/multirun/2023-01-26/21-12-55/1/.hydra/overrides.yaml"

print(OmegaConf.load(path)[0].split("=")[1])
