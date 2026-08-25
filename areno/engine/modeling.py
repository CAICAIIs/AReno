"""Shared model construction and optimizer helpers for worker components."""

from __future__ import annotations

from pathlib import Path

import torch

from areno.engine.config import EngineConfig
from areno.engine.optim import AdamW8bit, AdamWFP32Master
from areno.models.registry import build_model


def param_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    """Return `main_grad` when available, otherwise autograd `.grad`."""

    main_grad = getattr(param, "main_grad", None)
    if isinstance(main_grad, torch.Tensor):
        return main_grad
    return param.grad


def quantize_model_weights_fp8(model: torch.nn.Module) -> int:
    """Mark every TP-parallel linear weight as FP8 for the decode path.

    Calls ``mark_fp8_weight`` on the weight of each column/merged/row-parallel
    linear so ``_areno_linear_forward`` routes those through the FP8 W8A16
    kernel (RFC 0001). Decode/inference-only — the Triton FP8 kernel has no
    backward, so this must not be used in a training step. Returns the count.
    """
    from areno.accel.kernels.fp8_linear import mark_fp8_weight
    from areno.engine.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear

    count = 0
    for module in model.modules():
        if isinstance(module, (ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear)):
            weight = getattr(module, "weight", None)
            if weight is not None and getattr(weight, "ndim", 0) == 2 and weight.numel() > 0 and weight.is_cuda:
                mark_fp8_weight(weight)
                count += 1
    if count:
        import logging

        logging.getLogger("areno").warning("FP8 quantized %d linear weights (decode path)", count)
    return count


def build_model_on_device(config: EngineConfig, device: torch.device) -> torch.nn.Module:
    """Construct the model directly on `device` under the configured dtype."""

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(config.model.dtype)
        with torch.device(device), skip_torch_init(enabled=config.model_path is not None and not config.dummy_load):
            model = build_model(config.model)
        if config.dummy_load:
            _zero_model_parameters(model)
        return model
    finally:
        torch.set_default_dtype(old_dtype)


class skip_torch_init:
    """Temporarily replace expensive torch init functions with no-ops."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._saved = {}

    def __enter__(self):
        if not self.enabled:
            return self
        self._saved = {
            "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
            "uniform_": torch.nn.init.uniform_,
            "normal_": torch.nn.init.normal_,
        }
        torch.nn.init.kaiming_uniform_ = _noop_init
        torch.nn.init.uniform_ = _noop_init
        torch.nn.init.normal_ = _noop_init
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        for name, fn in self._saved.items():
            setattr(torch.nn.init, name, fn)


def _noop_init(tensor: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
    """Replacement for `torch.nn.init.*` that returns the tensor unchanged."""

    return tensor


def _zero_model_parameters(model: torch.nn.Module) -> None:
    """Initialize dummy-loaded models deterministically with finite zeros."""

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Recursively unwrap `torch.compile`'s `_orig_mod` indirection."""

    original = getattr(model, "_orig_mod", None)
    if original is None:
        return model
    return unwrap_model(original)


def canonical_model_path(path: str | None) -> str | None:
    """Resolve `path` to an absolute filesystem path for cache-key comparisons."""

    if path is None:
        return None
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError):
        return str(path)


def build_optimizer(params, optimizer_config, ctx, *, lr: float | None = None):
    """Construct the configured DP-sharded optimizer implementation."""

    optimizer_cls = AdamW8bit if optimizer_config.adam_8bit else AdamWFP32Master
    return optimizer_cls(
        params,
        lr=optimizer_config.lr if lr is None else float(lr),
        betas=optimizer_config.betas,
        weight_decay=optimizer_config.weight_decay,
        bucket_numel=optimizer_config.fp32_master_bucket_numel,
        dp_rank=ctx.dp_rank,
        dp_size=ctx.dp_size,
        dp_group=ctx.dp_group,
    )


def configure_multimodal_training(model: torch.nn.Module, optimizer_config, *, trainable: bool) -> None:
    """Apply optional model-defined media trainability and relative learning rates."""

    unwrapped = unwrap_model(model)
    configure = getattr(unwrapped, "configure_multimodal_training", None)
    requested = optimizer_config.unfreeze_multimodal_tower or optimizer_config.unfreeze_multimodal_projector
    if configure is None:
        if requested:
            raise ValueError(f"{type(unwrapped).__name__} does not expose configurable multimodal modules")
        return
    configure(
        unfreeze_tower=optimizer_config.unfreeze_multimodal_tower,
        unfreeze_projector=optimizer_config.unfreeze_multimodal_projector,
        tower_lr=optimizer_config.multimodal_tower_lr,
        projector_lr=optimizer_config.multimodal_projector_lr,
        base_lr=optimizer_config.lr,
        trainable=trainable,
    )
