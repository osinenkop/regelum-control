"""Contains blueprint of config for all optimizable objects."""
from dataclasses import dataclass, field
from typing import Optional, Any
import torch


@dataclass
class OptimizerConfig:
    """Base class for config of optimizable objects."""

    kind: str
    opt_method: Optional[Any] = None
    opt_options: dict = field(default_factory=lambda: {})
    log_options: dict = field(default_factory=lambda: {})
    config_options: dict = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        self.__dict__.update(**self.config_options)


torch_default_config = OptimizerConfig(
    kind="tensor",
    opt_options={"lr": 1e-3},
    opt_method=torch.optim.Adam,
    config_options={"n_epochs": 1},
)
casadi_default_config = OptimizerConfig(
    kind="symbolic",
    opt_options={"print_level": 0},
    log_options={"print_in": False, "print_out": False, "print_time": True},
    opt_method="ipopt",
)
scipy_default_config = OptimizerConfig(
    kind="numeric",
)
