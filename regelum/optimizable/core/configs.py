"""Contains blueprint of config for all optimizable objects."""

from typing import Optional, Any, Type, Dict
import torch
import casadi


class OptimizerConfig:
    """Base class for config of optimizable objects."""

    def __init__(
        self,
        kind: str,
        opt_method: Optional[Any] = None,
        opt_options: Optional[dict] = None,
        log_options: Optional[dict] = None,
        config_options: Optional[dict] = None,
    ) -> None:
        """Instantiate OptimizerConfig object.

        Args:
            kind (str): The optimization kind. Can be either of
                ["symbolic", "tensor", "numeric"]
            opt_method (Optional[Any], optional): What method to use.
                For `kind="tensor"` this is `torch.optim.Adam`, for
                instance. For kind="symbolic" this is "ipopt", defaults
                to None
            opt_options (Optional[dict], optional): Options to pass to
                the optimizer. For `kind="tensor"` this is `{"lr":
                0.001}`, for instance. For `kind="symbolic"` this is
                `{"print_level": 0}`, defaults to None
            log_options (Optional[dict], optional): Needed only for
                `kind="symbolic"`, defaults to None
            config_options (Optional[dict], optional): Other global
                options. `n_epochs`, `data_buffer_sampling_method`,
                `data_buffer_sampling_kwargs`, etc., defaults to None
        """
        self.kind = kind
        self.opt_method = opt_method
        self.opt_options = opt_options if opt_options is not None else {}
        self.log_options = log_options if log_options is not None else {}
        self.config_options = config_options if config_options is not None else {}


class TorchOptimizerConfig(OptimizerConfig):
    """Config for torch-based optimizers."""

    def __init__(
        self,
        n_epochs: int,
        data_buffer_iter_bathes_kwargs: Dict[str, Any],
        opt_method_kwargs: Dict[str, Any],
        opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        is_reinstantiate_optimizer: bool = True,
        n_epochs_per_constraint: Optional[int] = None,
    ) -> None:
        """Instantiate TorchOptimizerConfig object.

        Args:
            n_epochs (int.): How many epochs to use during optimization.
            data_buffer_iter_bathes_kwargs (Dict[str, Any]): kwargs for
                `DataBuffer.iter_batches`.
            opt_method_kwargs (Dict[str, Any]): What options to pass to
                the optimizer
            opt_method (Type[torch.optim.Optimizer], optional): What
                optimizer method to use, defaults to torch.optim.Adam
            is_reinstantiate_optimizer (bool, optional): Whether to
                reinstantiate optimizer every time `optimize()` is
                called, defaults to True
            n_epochs_per_constraint (Optional[int], optional): How many
                gradient steps to take to find a constraint feasible
                domain, defaults to None
        """
        super().__init__(
            kind="tensor",
            opt_method=opt_method,
            opt_options=opt_method_kwargs,
            log_options=None,
            config_options={
                "n_epochs": n_epochs,
                "data_buffer_sampling_method": "iter_batches",
                "data_buffer_sampling_kwargs": data_buffer_iter_bathes_kwargs,
                "is_reinstantiate_optimizer": is_reinstantiate_optimizer,
            }
            | (
                {
                    "constrained_optimization_policy": {
                        "is_activated": True,
                        "defaults": {
                            "n_epochs_per_constraint": n_epochs_per_constraint
                        },
                    }
                }
                if n_epochs_per_constraint is not None
                else {}
            ),
        )


class CasadiOptimizerConfig(OptimizerConfig):
    """Config for casadi-based optimizers."""

    def __init__(
        self,
        batch_size=1,
    ) -> None:
        """Instantiate CasadiOptimizerConfig object.

        Args:
            batch_size (int, optional): How many latest samples to use
                from `DataBuffer`, defaults to 1
        """
        super().__init__(
            kind="symbolic",
            opt_method="ipopt",
            opt_options={"print_level": 0},
            log_options={"print_in": False, "print_out": False, "print_time": False},
            config_options={
                "data_buffer_sampling_method": "sample_last",
                "data_buffer_sampling_kwargs": {
                    "n_samples": batch_size,
                    "dtype": casadi.DM,
                },
            },
        )


class ScipyOptimizerConfig(OptimizerConfig):
    def __init__(
        self,
        kind: str = "numeric",
        opt_method: Optional[Any] = "SLSQP",
        opt_options: Optional[Dict] = None,
        log_options: Optional[Dict] = None,
        config_options: Optional[Dict] = None,
    ) -> None:
        super().__init__(kind, opt_method, opt_options, config_options)
