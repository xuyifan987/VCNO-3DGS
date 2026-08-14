"""Reliability-conditioned per-primitive vehicle appearance network.

The network predicts one RGB color per Gaussian for the active camera.  It is
evaluated in PyTorch and feeds precomputed colors to the standard 2DGS
rasterizer, so the appearance representation is independent of custom neural
rasterization kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reliability_conditioned_parameters(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    context_gain: torch.Tensor | None = None,
    view_adapter: torch.Tensor | None = None,
    bias_adapter: torch.Tensor | None = None,
    hidden_gain: torch.Tensor | None = None,
    output_adapter: torch.Tensor | None = None,
    modulation: torch.Tensor | None = None,
    adapter_scale: float = 0.7,
    view_strength: float = 0.5,
    bias_strength: float = 0.25,
    output_strength: float = 0.15,
    view_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a gated residual adapter to batched per-Gaussian MLP weights.

    The view adapter is a rank-one update restricted to the three view-direction
    inputs.  Reliability controls the adapter magnitude but does not suppress the
    base appearance response.
    """
    if modulation is None or context_gain is None:
        return w1, b1, w2

    gate = modulation.to(dtype=w1.dtype, device=w1.device).clamp(0.0, 2.0)
    scale = max(float(adapter_scale), 0.0)

    row_gain = 1.0 + scale * gate * torch.tanh(context_gain)
    effective_w1 = w1 * row_gain.unsqueeze(-1)

    if view_adapter is not None:
        view_basis = torch.zeros((1, 1, w1.shape[-1]), dtype=w1.dtype, device=w1.device)
        view_start = max(int(view_start), 0)
        view_end = min(view_start + 3, w1.shape[-1])
        view_dims = max(view_end - view_start, 0)
        if view_dims > 0:
            view_basis[..., view_start:view_end] = 1.0 / view_dims
        effective_w1 = effective_w1 + (
            max(float(view_strength), 0.0)
            * gate.unsqueeze(-1)
            * torch.tanh(view_adapter).unsqueeze(-1)
            * view_basis
        )

    effective_b1 = b1
    if bias_adapter is not None:
        effective_b1 = b1 + (
            max(float(bias_strength), 0.0) * gate * torch.tanh(bias_adapter)
        )

    effective_w2 = w2
    if hidden_gain is not None:
        column_gain = 1.0 + scale * gate * torch.tanh(hidden_gain)
        effective_w2 = effective_w2 * column_gain.unsqueeze(1)

    if output_adapter is not None and output_strength > 0.0:
        local_scale = w2.detach().abs().mean(dim=(1, 2), keepdim=True).clamp_min(0.05)
        effective_w2 = effective_w2 + (
            max(float(output_strength), 0.0)
            * gate.unsqueeze(1)
            * local_scale
            * torch.tanh(output_adapter)
        )

    return effective_w1, effective_b1, effective_w2


def neural_vehicle_color(
    appearance_input: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    output_delta_scale: float = 1.5,
    output_mode: str = "sigmoid",
) -> torch.Tensor:
    """Evaluate the compact per-Gaussian MLP and return RGB in ``[0, 1]``."""
    if appearance_input.ndim != 2:
        raise ValueError("appearance_input must have shape [N, C]")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("w1 and w2 must be batched matrices")
    if appearance_input.shape[0] != w1.shape[0] or w1.shape[0] != w2.shape[0]:
        raise ValueError("all appearance tensors must share the Gaussian dimension")
    if appearance_input.shape[1] != w1.shape[2]:
        raise ValueError("appearance input dimension does not match w1")

    hidden = torch.bmm(w1, appearance_input.unsqueeze(-1)).squeeze(-1) + b1
    hidden = F.silu(hidden)
    dynamic_delta = torch.bmm(w2, hidden.unsqueeze(-1)).squeeze(-1)
    if output_mode == "sigmoid":
        return torch.sigmoid(dynamic_delta + b2)
    delta_scale = max(float(output_delta_scale), 1e-6)
    logits = b2 + delta_scale * torch.tanh(dynamic_delta)
    return torch.sigmoid(logits)
