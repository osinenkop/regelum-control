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

        :param kind: The optimization kind. Can be either of ["symbolic", "tensor", "numeric"]
        :type kind: str
        :param opt_method: What method to use. For `kind="tensor"` this is `torch.optim.Adam`, for instance. For kind="symbolic" this is "ipopt", defaults to None
        :type opt_method: Optional[Any], optional
        :param opt_options: Options to pass to the optimizer. For `kind="tensor"` this is `{"lr": 0.001}`, for instance. For `kind="symbolic"` this is `{"print_level": 0}`, defaults to None
        :type opt_options: Optional[dict], optional
        :param log_options: Needed only for `kind="symbolic"`, defaults to None
        :type log_options: Optional[dict], optional
        :param config_options: Other global options. `n_epochs`, `data_buffer_sampling_method`, `data_buffer_sampling_kwargs`, etc., defaults to None
        :type config_options: Optional[dict], optional
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

        :param n_epochs: How many epochs to use during optimization.
        :type n_epochs: int.
        :param data_buffer_iter_bathes_kwargs: kwargs for `DataBuffer.iter_batches`.
        :type data_buffer_iter_bathes_kwargs: Dict[str, Any]
        :param opt_method_kwargs: What options to pass to the optimizer
        :type opt_method_kwargs: Dict[str, Any]
        :param opt_method: What optimizer method to use, defaults to torch.optim.Adam
        :type opt_method: Type[torch.optim.Optimizer], optional
        :param is_reinstantiate_optimizer: Whether to reinstantiate optimizer every time `optimize()` is called, defaults to True
        :type is_reinstantiate_optimizer: bool, optional
        :param n_epochs_per_constraint: How many gradient steps to take to find a constraint feasible domain, defaults to None
        :type n_epochs_per_constraint: Optional[int], optional
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

        :param batch_size: How many latest samples to use from `DataBuffer`, defaults to 1
        :type batch_size: int, optional
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
