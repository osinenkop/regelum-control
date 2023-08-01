"""Contains blueprint of config for all optimizable objects."""
from dataclasses import dataclass, field
from typing import Optional, Any, List


@dataclass
class OptimizerConfig:
    """Base class for config of optimizable objects."""

    kind: str
    callback_target_events: Optional[List[str]] = None
    opt_method: Optional[Any] = None
    opt_options: dict = field(default_factory=lambda: {})
    log_options: dict = field(default_factory=lambda: {})
    config_options: dict = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        self.__dict__.update(**self.config_options)


torch_default_config = OptimizerConfig(
    kind="tensor",
    opt_options={"lr": 1e-3},
    config_options={"batch_size": 500, "shuffle": False, "iterations": 30},
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
