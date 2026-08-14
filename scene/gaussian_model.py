#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
import math
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.grid_sdf import TrilinearInterpolation
from utils.neural_vehicle_appearance import (
    neural_vehicle_color,
    reliability_conditioned_parameters,
)

class GaussianModel:

    APPEARANCE_MODEL_VERSION = "reliability_conditioned_mlp_v3_bounded_dynamic"

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, appearance_backend: str = "torch_mlp"):
        self.hidden_neuron = 6
        self.output_dim = 3
        # View direction, vehicle-local Gaussian coordinate, and intrinsic 2D scale context.
        self.input_dim = 3 + 3 + 2
        self.appearance_backend = str(appearance_backend)

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._W1 = torch.empty(0)
        self._b1 = torch.empty(0)
        self._W2 = torch.empty(0)
        self._b2 = torch.empty(0)
        self._gabor_freq_delta = torch.empty(0)
        self._gabor_view_delta = torch.empty(0)
        self._gabor_phase_delta = torch.empty(0)
        self._gabor_amp_delta = torch.empty(0)
        self._gabor_w2_residual = torch.empty(0)
        self._gabor_reliability = torch.empty(0)
        self._gabor_detail_confidence = torch.empty(0)
        self._gabor_tsdf_confidence = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.tsdf = None
        self.tsdf_tensor = None
        self.vol_origin = None
        self.vol_origin_tensor = None
        self.voxel_size = None
        self.tsdf_trunc = None
        self.interp = TrilinearInterpolation()
        self.adaptive_gabor_enabled = False
        self.adaptive_gabor_strength = 0.75
        self.adaptive_gabor_max_log_scale = 0.70
        self.adaptive_gabor_view_strength = 0.50
        self.adaptive_gabor_phase_strength = 0.25
        self.adaptive_gabor_residual_strength = 0.15
        self.adaptive_gabor_residual_detail_power = 0.0
        self.adaptive_gabor_residual_scale_power = 0.0
        self.adaptive_gabor_residual_scale_quantile = 0.5
        self.adaptive_gabor_residual_scale_min_gate = 0.0
        self.adaptive_gabor_detail_weight = 0.75
        self.adaptive_gabor_min_confidence = 0.15
        self.adaptive_gabor_tsdf_weight = 0.50
        self.adaptive_gabor_tsdf_default_confidence = 0.0
        self.adaptive_gabor_start_iter = 1500
        self.adaptive_gabor_warmup = 1000
        self.adaptive_gabor_preserve_base = False
        self.neural_output_mode = "sigmoid"
        self.neural_output_delta_scale = 1.5
        self.adaptive_gabor_current_iteration = 0
        self.setup_functions()

    def configure_adaptive_gabor(self, training_args=None, enabled=None):
        if training_args is not None:
            enabled = getattr(training_args, "car_fusion_adaptive_gabor", self.adaptive_gabor_enabled)
            if hasattr(training_args, "car_fusion"):
                enabled = bool(enabled and getattr(training_args, "car_fusion"))
            self.adaptive_gabor_strength = float(getattr(training_args, "car_fusion_adaptive_gabor_strength", self.adaptive_gabor_strength))
            self.adaptive_gabor_max_log_scale = float(getattr(training_args, "car_fusion_adaptive_gabor_max_log_scale", self.adaptive_gabor_max_log_scale))
            self.adaptive_gabor_view_strength = float(getattr(training_args, "car_fusion_adaptive_gabor_view_strength", self.adaptive_gabor_view_strength))
            self.adaptive_gabor_phase_strength = float(getattr(training_args, "car_fusion_adaptive_gabor_phase_strength", self.adaptive_gabor_phase_strength))
            self.adaptive_gabor_residual_strength = float(getattr(training_args, "car_fusion_adaptive_gabor_residual_strength", self.adaptive_gabor_residual_strength))
            self.adaptive_gabor_residual_detail_power = float(getattr(training_args, "car_fusion_adaptive_gabor_residual_detail_power", self.adaptive_gabor_residual_detail_power))
            self.adaptive_gabor_residual_scale_power = float(getattr(training_args, "car_fusion_adaptive_gabor_residual_scale_power", self.adaptive_gabor_residual_scale_power))
            self.adaptive_gabor_residual_scale_quantile = float(getattr(training_args, "car_fusion_adaptive_gabor_residual_scale_quantile", self.adaptive_gabor_residual_scale_quantile))
            self.adaptive_gabor_residual_scale_min_gate = float(getattr(training_args, "car_fusion_adaptive_gabor_residual_scale_min_gate", self.adaptive_gabor_residual_scale_min_gate))
            self.adaptive_gabor_detail_weight = float(getattr(training_args, "car_fusion_adaptive_gabor_detail_weight", self.adaptive_gabor_detail_weight))
            self.adaptive_gabor_min_confidence = float(getattr(training_args, "car_fusion_adaptive_gabor_min_confidence", self.adaptive_gabor_min_confidence))
            self.adaptive_gabor_tsdf_weight = float(getattr(training_args, "car_fusion_adaptive_gabor_tsdf_weight", self.adaptive_gabor_tsdf_weight))
            self.adaptive_gabor_tsdf_default_confidence = float(getattr(training_args, "car_fusion_adaptive_gabor_tsdf_default_confidence", self.adaptive_gabor_tsdf_default_confidence))
            self.adaptive_gabor_start_iter = int(getattr(training_args, "car_fusion_adaptive_gabor_start_iter", self.adaptive_gabor_start_iter))
            self.adaptive_gabor_warmup = int(getattr(training_args, "car_fusion_adaptive_gabor_warmup", self.adaptive_gabor_warmup))
            self.adaptive_gabor_preserve_base = bool(getattr(training_args, "car_fusion_adaptive_gabor_preserve_base", self.adaptive_gabor_preserve_base))
            self.neural_output_mode = str(getattr(training_args, "car_fusion_neural_output_mode", self.neural_output_mode))
            self.neural_output_delta_scale = float(getattr(training_args, "car_fusion_neural_output_delta_scale", self.neural_output_delta_scale))
        if enabled is not None:
            self.adaptive_gabor_enabled = bool(enabled)
        self._ensure_adaptive_gabor_state()

    def set_adaptive_gabor_iteration(self, iteration):
        self.adaptive_gabor_current_iteration = int(iteration)

    def _ensure_adaptive_gabor_state(self):
        if self._xyz.numel() == 0:
            return
        device = self._xyz.device
        n_points = self._xyz.shape[0]
        if self._gabor_freq_delta.numel() == 0 or self._gabor_freq_delta.shape[0] != n_points:
            self._gabor_freq_delta = nn.Parameter(torch.zeros((n_points, self.hidden_neuron), device=device).requires_grad_(True))
        elif not isinstance(self._gabor_freq_delta, nn.Parameter) or self._gabor_freq_delta.device != device:
            self._gabor_freq_delta = nn.Parameter(self._gabor_freq_delta.detach().to(device).requires_grad_(True))
        if self._gabor_view_delta.numel() == 0 or self._gabor_view_delta.shape[0] != n_points:
            self._gabor_view_delta = nn.Parameter(torch.zeros((n_points, self.hidden_neuron), device=device).requires_grad_(True))
        elif not isinstance(self._gabor_view_delta, nn.Parameter) or self._gabor_view_delta.device != device:
            self._gabor_view_delta = nn.Parameter(self._gabor_view_delta.detach().to(device).requires_grad_(True))
        if self._gabor_phase_delta.numel() == 0 or self._gabor_phase_delta.shape[0] != n_points:
            self._gabor_phase_delta = nn.Parameter(torch.zeros((n_points, self.hidden_neuron), device=device).requires_grad_(True))
        elif not isinstance(self._gabor_phase_delta, nn.Parameter) or self._gabor_phase_delta.device != device:
            self._gabor_phase_delta = nn.Parameter(self._gabor_phase_delta.detach().to(device).requires_grad_(True))
        if self._gabor_amp_delta.numel() == 0 or self._gabor_amp_delta.shape[0] != n_points:
            self._gabor_amp_delta = nn.Parameter(torch.zeros((n_points, self.hidden_neuron), device=device).requires_grad_(True))
        elif not isinstance(self._gabor_amp_delta, nn.Parameter) or self._gabor_amp_delta.device != device:
            self._gabor_amp_delta = nn.Parameter(self._gabor_amp_delta.detach().to(device).requires_grad_(True))
        if self._gabor_w2_residual.numel() == 0 or self._gabor_w2_residual.shape[0] != n_points:
            self._gabor_w2_residual = nn.Parameter(torch.zeros((n_points, self.output_dim, self.hidden_neuron), device=device).requires_grad_(True))
        elif not isinstance(self._gabor_w2_residual, nn.Parameter) or self._gabor_w2_residual.device != device:
            self._gabor_w2_residual = nn.Parameter(self._gabor_w2_residual.detach().to(device).requires_grad_(True))
        if self._gabor_reliability.numel() == 0 or self._gabor_reliability.shape[0] != n_points:
            self._gabor_reliability = torch.full((n_points, 1), 0.5, device=device)
        if self._gabor_detail_confidence.numel() == 0 or self._gabor_detail_confidence.shape[0] != n_points:
            self._gabor_detail_confidence = torch.full((n_points, 1), 0.5, device=device)
        if self._gabor_tsdf_confidence.numel() == 0 or self._gabor_tsdf_confidence.shape[0] != n_points:
            default_tsdf_confidence = min(max(float(self.adaptive_gabor_tsdf_default_confidence), 0.0), 1.0)
            self._gabor_tsdf_confidence = torch.full((n_points, 1), default_tsdf_confidence, device=device)

    def _adaptive_gabor_confidence(self):
        self._ensure_adaptive_gabor_state()
        if self._xyz.numel() == 0:
            return torch.empty((0, 1), device="cuda")
        reliability = self._gabor_reliability.to(self._xyz.device).clamp(0.0, 1.0)
        detail_confidence = self._gabor_detail_confidence.to(self._xyz.device).clamp(0.0, 1.0)
        tsdf_confidence = self._gabor_tsdf_confidence.to(self._xyz.device).clamp(0.0, 1.0)
        detail_weight = min(max(float(self.adaptive_gabor_detail_weight), 0.0), 1.0)
        tsdf_weight = min(max(float(self.adaptive_gabor_tsdf_weight), 0.0), 1.0)
        # TSDF is self-generated and can be conservative early; use it as a surface-consistency boost
        # without letting low TSDF confidence suppress photometrically reliable vehicle details.
        detail_gate = (1.0 - detail_weight) + detail_weight * detail_confidence
        reliable_detail = reliability * detail_gate
        return (reliable_detail + tsdf_weight * tsdf_confidence * (1.0 - reliable_detail)).clamp(0.0, 1.0)

    def _adaptive_gabor_gate(self):
        confidence = self._adaptive_gabor_confidence()
        min_conf = min(max(float(self.adaptive_gabor_min_confidence), 0.0), 0.99)
        return ((confidence - min_conf) / max(1.0 - min_conf, 1e-6)).clamp(0.0, 1.0)

    def _adaptive_gabor_modulation(self):
        gate = self._adaptive_gabor_gate()
        strength = max(float(self.adaptive_gabor_strength), 0.0)
        schedule = self._adaptive_gabor_schedule()
        return (strength * schedule * gate).clamp(0.0, 2.0)

    def _adaptive_gabor_schedule(self):
        start_iter = max(int(self.adaptive_gabor_start_iter), 0)
        warmup = max(int(self.adaptive_gabor_warmup), 1)
        if self.adaptive_gabor_current_iteration <= start_iter:
            return 0.0
        return min((self.adaptive_gabor_current_iteration - start_iter) / warmup, 1.0)

    def get_adaptive_gabor_stats(self):
        if not self.adaptive_gabor_enabled or self._xyz.numel() == 0:
            return {
                "confidence": 0.0,
                "gate": 0.0,
                "schedule": 0.0,
                "detail": 0.0,
                "context_gain": 1.0,
                "view_adapter": 0.0,
                "bias_adapter": 0.0,
                "hidden_gain": 1.0,
                "output_adapter": 0.0,
                "output_route_gate": 1.0,
            }
        with torch.no_grad():
            modulation = self._adaptive_gabor_modulation()
            adapter_scale = max(float(self.adaptive_gabor_max_log_scale), 0.0)
            context_gain = 1.0 + adapter_scale * modulation * torch.tanh(self._gabor_freq_delta)
            view_adapter = max(float(self.adaptive_gabor_view_strength), 0.0) * modulation * torch.tanh(self._gabor_view_delta)
            bias_adapter = max(float(self.adaptive_gabor_phase_strength), 0.0) * modulation * torch.tanh(self._gabor_phase_delta)
            hidden_gain = 1.0 + adapter_scale * modulation * torch.tanh(self._gabor_amp_delta)
            output_route_gate = self._adaptive_gabor_residual_route_gate()
            output_adapter = (
                max(float(self.adaptive_gabor_residual_strength), 0.0)
                * modulation.unsqueeze(1)
                * output_route_gate.unsqueeze(1)
                * torch.tanh(self._gabor_w2_residual)
            )
            return {
                "confidence": float(self._adaptive_gabor_confidence().mean().item()),
                "gate": float(self._adaptive_gabor_gate().mean().item()),
                "schedule": float(self._adaptive_gabor_schedule()),
                "detail": float(self._gabor_detail_confidence.mean().item()),
                "context_gain": float(context_gain.mean().item()),
                "view_adapter": float(view_adapter.abs().mean().item()),
                "bias_adapter": float(bias_adapter.abs().mean().item()),
                "hidden_gain": float(hidden_gain.mean().item()),
                "output_adapter": float(output_adapter.abs().mean().item()),
                "output_route_gate": float(output_route_gate.mean().item()),
            }

    def adaptive_gabor_regularization(self):
        if not self.adaptive_gabor_enabled or self._xyz.numel() == 0:
            device = self._xyz.device if self._xyz.numel() > 0 else torch.device("cuda")
            return torch.zeros((), device=device)
        self._ensure_adaptive_gabor_state()
        gate = self._adaptive_gabor_gate().detach()
        context_penalty = torch.tanh(self._gabor_freq_delta).pow(2).mean(dim=1, keepdim=True)
        view_penalty = torch.tanh(self._gabor_view_delta).pow(2).mean(dim=1, keepdim=True)
        bias_penalty = torch.tanh(self._gabor_phase_delta).pow(2).mean(dim=1, keepdim=True)
        hidden_penalty = torch.tanh(self._gabor_amp_delta).pow(2).mean(dim=1, keepdim=True)
        output_penalty = torch.tanh(self._gabor_w2_residual).pow(2).mean(dim=(1, 2)).unsqueeze(-1)
        adapter_penalty = context_penalty + view_penalty + 0.5 * bias_penalty + hidden_penalty + output_penalty
        return (adapter_penalty * (0.25 + gate)).mean()

    def _adaptive_gabor_residual_route_gate(self):
        return self._adaptive_gabor_residual_detail_gate() * self._adaptive_gabor_residual_scale_gate()

    def _adaptive_gabor_residual_detail_gate(self):
        if self._xyz.numel() == 0:
            return torch.empty((0, 1), device="cuda")
        power = max(float(self.adaptive_gabor_residual_detail_power), 0.0)
        if power <= 0.0:
            return torch.ones((self._xyz.shape[0], 1), device=self._xyz.device)
        self._ensure_adaptive_gabor_state()
        detail = self._gabor_detail_confidence.to(self._xyz.device).clamp(0.0, 1.0)
        return detail.pow(power).clamp(0.0, 1.0)

    def _adaptive_gabor_residual_scale_gate(self):
        if self._xyz.numel() == 0:
            return torch.empty((0, 1), device="cuda")
        power = max(float(self.adaptive_gabor_residual_scale_power), 0.0)
        if power <= 0.0:
            return torch.ones((self._xyz.shape[0], 1), device=self._xyz.device)
        scale = self.get_scaling.detach().amax(dim=1, keepdim=True).clamp_min(1e-6)
        quantile = min(max(float(self.adaptive_gabor_residual_scale_quantile), 0.05), 0.95)
        reference = torch.quantile(scale.flatten(), quantile).detach().clamp_min(1e-6)
        gate = (reference / scale).clamp(0.0, 1.0).pow(power)
        min_gate = min(max(float(self.adaptive_gabor_residual_scale_min_gate), 0.0), 1.0)
        if min_gate > 0.0:
            gate = min_gate + (1.0 - min_gate) * gate
        return gate.clamp(0.0, 1.0)

    @torch.no_grad()
    def update_adaptive_gabor_reliability(self, projected_reliability, momentum=0.90):
        if not self.adaptive_gabor_enabled or projected_reliability is None or self._xyz.numel() == 0:
            return {"mean": 0.0}
        self._ensure_adaptive_gabor_state()
        reliability = projected_reliability.detach().reshape(-1, 1).to(self._xyz.device).clamp(0.0, 1.0)
        if reliability.shape[0] != self._xyz.shape[0]:
            n = min(reliability.shape[0], self._xyz.shape[0])
            reliability_full = self._gabor_reliability.clone()
            reliability_full[:n] = reliability[:n]
            reliability = reliability_full
        momentum = min(max(float(momentum), 0.0), 0.999)
        self._gabor_reliability.mul_(momentum).add_(reliability * (1.0 - momentum))
        return {"mean": float(self._gabor_reliability.mean().item())}

    @torch.no_grad()
    def update_adaptive_gabor_detail(self, projected_detail, momentum=0.90):
        if not self.adaptive_gabor_enabled or projected_detail is None or self._xyz.numel() == 0:
            return {"mean": 0.0}
        self._ensure_adaptive_gabor_state()
        detail = projected_detail.detach().reshape(-1, 1).to(self._xyz.device).clamp(0.0, 1.0)
        if detail.shape[0] != self._xyz.shape[0]:
            n = min(detail.shape[0], self._xyz.shape[0])
            detail_full = self._gabor_detail_confidence.clone()
            detail_full[:n] = detail[:n]
            detail = detail_full
        momentum = min(max(float(momentum), 0.0), 0.999)
        self._gabor_detail_confidence.mul_(momentum).add_(detail * (1.0 - momentum))
        return {"mean": float(self._gabor_detail_confidence.mean().item())}

    @torch.no_grad()
    def refresh_adaptive_gabor_tsdf_confidence(self):
        if not self.adaptive_gabor_enabled or not self.has_tsdf() or self._xyz.numel() == 0:
            return {"mean": 0.0}
        self._ensure_adaptive_gabor_state()
        sdf = self.get_sdf(self.get_xyz)
        if sdf is None or sdf.numel() == 0:
            return {"mean": 0.0}
        trunc = max(float(self.tsdf_trunc if self.tsdf_trunc is not None else 1.0), 1e-6)
        confidence = torch.exp(-sdf.abs() / trunc).clamp(0.0, 1.0)
        confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
        self._gabor_tsdf_confidence = confidence.detach()
        return {"mean": float(self._gabor_tsdf_confidence.mean().item())}

    def set_tsdf(self, tsdf, vol_origin, voxel_size, tsdf_trunc, interp=None):
        self.tsdf = tsdf.astype(np.float32) if isinstance(tsdf, np.ndarray) else np.asarray(tsdf, dtype=np.float32)
        self.vol_origin = (
            vol_origin.astype(np.float32)
            if isinstance(vol_origin, np.ndarray)
            else np.asarray(vol_origin, dtype=np.float32)
        )
        self.voxel_size = float(voxel_size)
        self.tsdf_trunc = float(tsdf_trunc)
        self.interp = interp if interp is not None else TrilinearInterpolation()
        self.tsdf_tensor = None
        self.vol_origin_tensor = None
        self._sync_tsdf_device()

    def has_tsdf(self):
        return self.tsdf is not None and self.vol_origin is not None and self.voxel_size is not None

    def release_tsdf_gpu_cache(self):
        self.tsdf_tensor = None
        self.vol_origin_tensor = None

    def _sync_tsdf_device(self):
        if not self.has_tsdf():
            return
        device = self._xyz.device if self._xyz.numel() > 0 else torch.device("cuda")
        if self.tsdf_tensor is None or self.tsdf_tensor.device != device:
            self.tsdf_tensor = torch.from_numpy(self.tsdf[None, None]).float().to(device)
        if self.vol_origin_tensor is None or self.vol_origin_tensor.device != device:
            self.vol_origin_tensor = torch.from_numpy(self.vol_origin).float().to(device)

    def _tsdf_sampling_grid(self, points):
        self._sync_tsdf_device()
        if not self.has_tsdf():
            return None
        pts = (points - self.vol_origin_tensor) / self.voxel_size
        dim_x, dim_y, dim_z = self.tsdf.shape
        x = 2.0 * pts[:, 0] / max(dim_x - 1, 1) - 1.0
        y = 2.0 * pts[:, 1] / max(dim_y - 1, 1) - 1.0
        z = 2.0 * pts[:, 2] / max(dim_z - 1, 1) - 1.0
        # TrilinearInterpolation follows PyTorch grid_sample's x/y/z order,
        # so the TSDF tensor layout [X, Y, Z] is queried as [z, y, x].
        return torch.stack((z, y, x), dim=-1).view(1, -1, 1, 3)

    @torch.no_grad()
    def get_sdf(self, points=None):
        if not self.has_tsdf():
            return None
        points = self.get_xyz if points is None else points
        if points.numel() == 0:
            return torch.empty((0, 1), device=points.device)
        grid = self._tsdf_sampling_grid(points.detach())
        return self.interp(self.tsdf_tensor, grid).reshape(-1, 1)

    @torch.no_grad()
    def compute_sdf_gradient(self, points=None, h=None):
        if not self.has_tsdf():
            return None, None
        points = self.get_xyz if points is None else points
        h = float(h if h is not None else max(self.voxel_size, 1e-4))
        grad = torch.zeros_like(points)
        sdf = self.get_sdf(points)
        for axis in range(3):
            delta = torch.zeros_like(points)
            delta[:, axis] = h
            sdf_plus = self.get_sdf(points + delta)
            sdf_minus = self.get_sdf(points - delta)
            grad[:, axis] = ((sdf_plus - sdf_minus) / (2.0 * h)).squeeze(-1)
        return grad, sdf

    @torch.no_grad()
    def tsdf_prune_and_pull(
        self,
        pull_weight=1e-4,
        prune=False,
        outside=0.999,
        opacity_limit=0.10,
        max_pull=0.002,
        confidence_gate=False,
        min_reliability=0.10,
        min_tsdf_confidence=0.02,
        gate_power=1.0,
        detail_suppression=0.0,
    ):
        if not self.has_tsdf() or self.get_xyz.numel() == 0:
            return {"pruned": 0, "pulled": 0, "mean_abs_sdf": 0.0, "pull_gate": 1.0}

        sdf = self.get_sdf(self.get_xyz)
        pruned = 0
        if prune and sdf is not None and sdf.numel() > 0:
            outside_mask = (sdf.abs().squeeze(-1) >= outside) & (self.get_opacity.squeeze(-1) < opacity_limit)
            if outside_mask.any():
                pruned = int(outside_mask.sum().item())
                self.prune_points(outside_mask)
                sdf = self.get_sdf(self.get_xyz)

        grad, sdf = self.compute_sdf_gradient(self.get_xyz)
        if grad is None or sdf is None or sdf.numel() == 0:
            return {"pruned": pruned, "pulled": 0, "mean_abs_sdf": 0.0, "pull_gate": 1.0}

        valid = torch.isfinite(sdf.squeeze(-1)) & (sdf.abs().squeeze(-1) < outside)
        pull_gate = torch.ones_like(sdf)
        if confidence_gate:
            n_points = self.get_xyz.shape[0]
            if self._gabor_reliability.numel() == n_points:
                reliability = self._gabor_reliability.to(self._xyz.device).clamp(0.0, 1.0)
                min_rel = min(max(float(min_reliability), 0.0), 0.999)
                reliability_gate = ((reliability - min_rel) / max(1.0 - min_rel, 1e-6)).clamp(0.0, 1.0)
                pull_gate = pull_gate * reliability_gate
            if self._gabor_tsdf_confidence.numel() == n_points:
                tsdf_confidence = self._gabor_tsdf_confidence.to(self._xyz.device).clamp(0.0, 1.0)
                min_conf = min(max(float(min_tsdf_confidence), 0.0), 0.999)
                tsdf_gate = ((tsdf_confidence - min_conf) / max(1.0 - min_conf, 1e-6)).clamp(0.0, 1.0)
                pull_gate = pull_gate * tsdf_gate
            if self._gabor_detail_confidence.numel() == n_points and detail_suppression > 0:
                detail = self._gabor_detail_confidence.to(self._xyz.device).clamp(0.0, 1.0)
                pull_gate = pull_gate * (1.0 - detail).clamp(0.0, 1.0).pow(float(detail_suppression))
            if gate_power != 1.0:
                pull_gate = pull_gate.clamp(0.0, 1.0).pow(float(gate_power))
            valid = valid & (pull_gate.squeeze(-1) > 0.0)
        if valid.any() and pull_weight > 0:
            direction = F.normalize(grad, dim=-1)
            step = torch.clamp(sdf * float(pull_weight) * pull_gate, min=-float(max_pull), max=float(max_pull))
            self._xyz.data[valid] -= step[valid] * direction[valid]
        return {
            "pruned": pruned,
            "pulled": int(valid.sum().item()),
            "mean_abs_sdf": float(sdf.abs().mean().item()),
            "pull_gate": float(pull_gate[valid].mean().item()) if valid.any() else 0.0,
        }

    def capture(self):
        adaptive_gabor_state = {
            "appearance_model": self.APPEARANCE_MODEL_VERSION,
            "freq_delta": self._gabor_freq_delta,
            "view_delta": self._gabor_view_delta,
            "phase_delta": self._gabor_phase_delta,
            "amp_delta": self._gabor_amp_delta,
            "w2_residual": self._gabor_w2_residual,
            "reliability": self._gabor_reliability,
            "detail_confidence": self._gabor_detail_confidence,
            "tsdf_confidence": self._gabor_tsdf_confidence,
            "enabled": self.adaptive_gabor_enabled,
            "strength": self.adaptive_gabor_strength,
            "max_log_scale": self.adaptive_gabor_max_log_scale,
            "view_strength": self.adaptive_gabor_view_strength,
            "phase_strength": self.adaptive_gabor_phase_strength,
            "residual_strength": self.adaptive_gabor_residual_strength,
            "residual_detail_power": self.adaptive_gabor_residual_detail_power,
            "residual_scale_power": self.adaptive_gabor_residual_scale_power,
            "residual_scale_quantile": self.adaptive_gabor_residual_scale_quantile,
            "residual_scale_min_gate": self.adaptive_gabor_residual_scale_min_gate,
            "detail_weight": self.adaptive_gabor_detail_weight,
            "min_confidence": self.adaptive_gabor_min_confidence,
            "tsdf_weight": self.adaptive_gabor_tsdf_weight,
            "tsdf_default_confidence": self.adaptive_gabor_tsdf_default_confidence,
            "start_iter": self.adaptive_gabor_start_iter,
            "warmup": self.adaptive_gabor_warmup,
            "preserve_base": self.adaptive_gabor_preserve_base,
            "neural_output_mode": self.neural_output_mode,
            "neural_output_delta_scale": self.neural_output_delta_scale,
            "current_iteration": self.adaptive_gabor_current_iteration,
        }
        return (
            self.active_sh_degree,
            self._xyz,
            self._W1,
            self._b1,
            self._W2,
            self._b2,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            adaptive_gabor_state,
        )
    
    def restore(self, model_args, training_args):
        adaptive_gabor_state = model_args[14] if len(model_args) > 14 else {}
        checkpoint_appearance = adaptive_gabor_state.get("appearance_model") if adaptive_gabor_state else None
        if checkpoint_appearance != self.APPEARANCE_MODEL_VERSION:
            raise ValueError(
                "This checkpoint predates the reliability-conditioned appearance MLP and "
                "cannot be resumed safely. Start a new run with the current model."
            )
        (self.active_sh_degree,
        self._xyz,
        self._W1,
        self._b1,
        self._W2,
        self._b2,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args[:14]
        if adaptive_gabor_state:
            self._gabor_freq_delta = adaptive_gabor_state.get("freq_delta", torch.empty(0))
            self._gabor_view_delta = adaptive_gabor_state.get("view_delta", torch.empty(0))
            self._gabor_phase_delta = adaptive_gabor_state.get("phase_delta", torch.empty(0))
            self._gabor_amp_delta = adaptive_gabor_state.get("amp_delta", torch.empty(0))
            self._gabor_w2_residual = adaptive_gabor_state.get("w2_residual", torch.empty(0))
            self._gabor_reliability = adaptive_gabor_state.get("reliability", torch.empty(0))
            self._gabor_detail_confidence = adaptive_gabor_state.get("detail_confidence", torch.empty(0))
            self._gabor_tsdf_confidence = adaptive_gabor_state.get("tsdf_confidence", torch.empty(0))
            self.adaptive_gabor_enabled = bool(adaptive_gabor_state.get("enabled", self.adaptive_gabor_enabled))
            self.adaptive_gabor_strength = float(adaptive_gabor_state.get("strength", self.adaptive_gabor_strength))
            self.adaptive_gabor_max_log_scale = float(adaptive_gabor_state.get("max_log_scale", self.adaptive_gabor_max_log_scale))
            self.adaptive_gabor_view_strength = float(adaptive_gabor_state.get("view_strength", self.adaptive_gabor_view_strength))
            self.adaptive_gabor_phase_strength = float(adaptive_gabor_state.get("phase_strength", self.adaptive_gabor_phase_strength))
            self.adaptive_gabor_residual_strength = float(adaptive_gabor_state.get("residual_strength", self.adaptive_gabor_residual_strength))
            self.adaptive_gabor_residual_detail_power = float(adaptive_gabor_state.get("residual_detail_power", self.adaptive_gabor_residual_detail_power))
            self.adaptive_gabor_residual_scale_power = float(adaptive_gabor_state.get("residual_scale_power", self.adaptive_gabor_residual_scale_power))
            self.adaptive_gabor_residual_scale_quantile = float(adaptive_gabor_state.get("residual_scale_quantile", self.adaptive_gabor_residual_scale_quantile))
            self.adaptive_gabor_residual_scale_min_gate = float(adaptive_gabor_state.get("residual_scale_min_gate", self.adaptive_gabor_residual_scale_min_gate))
            self.adaptive_gabor_detail_weight = float(adaptive_gabor_state.get("detail_weight", self.adaptive_gabor_detail_weight))
            self.adaptive_gabor_min_confidence = float(adaptive_gabor_state.get("min_confidence", self.adaptive_gabor_min_confidence))
            self.adaptive_gabor_tsdf_weight = float(adaptive_gabor_state.get("tsdf_weight", self.adaptive_gabor_tsdf_weight))
            self.adaptive_gabor_tsdf_default_confidence = float(adaptive_gabor_state.get("tsdf_default_confidence", self.adaptive_gabor_tsdf_default_confidence))
            self.adaptive_gabor_start_iter = int(adaptive_gabor_state.get("start_iter", self.adaptive_gabor_start_iter))
            self.adaptive_gabor_warmup = int(adaptive_gabor_state.get("warmup", self.adaptive_gabor_warmup))
            self.adaptive_gabor_preserve_base = bool(adaptive_gabor_state.get("preserve_base", self.adaptive_gabor_preserve_base))
            self.neural_output_mode = str(adaptive_gabor_state.get("neural_output_mode", self.neural_output_mode))
            self.neural_output_delta_scale = float(adaptive_gabor_state.get("neural_output_delta_scale", self.neural_output_delta_scale))
            self.adaptive_gabor_current_iteration = int(adaptive_gabor_state.get("current_iteration", self.adaptive_gabor_current_iteration))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as exc:
            print(f"Warning: optimizer state was not restored because parameter groups changed: {exc}")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_layer_1_weight(self):
        effective_w1, effective_b1, _ = self._conditioned_appearance_parameters()
        return effective_w1, effective_b1
    
    @property
    def get_layer_2_weight(self):
        _, _, effective_w2 = self._conditioned_appearance_parameters()
        return effective_w2, self._b2

    def _vehicle_local_coordinates(self):
        if self._xyz.numel() == 0:
            return torch.empty((0, 3), device=self._xyz.device)
        xyz = self._xyz.detach()
        xyz_min = xyz.amin(dim=0, keepdim=True)
        xyz_max = xyz.amax(dim=0, keepdim=True)
        center = 0.5 * (xyz_min + xyz_max)
        extent = (xyz_max - xyz_min).amax(dim=1, keepdim=True).clamp_min(1e-6)
        return ((xyz - center) / extent).clamp(-1.0, 1.0)

    def _conditioned_appearance_parameters(self):
        """Return the reliability-conditioned MLP parameters.

        The legacy storage names are retained for checkpoint compatibility, but
        they now parameterize a generic residual adapter rather than frequency,
        phase, or Gabor-kernel transformations.
        """
        if not self.adaptive_gabor_enabled:
            return self._W1, self._b1, self._W2
        self._ensure_adaptive_gabor_state()
        output_adapter = self._gabor_w2_residual * self._adaptive_gabor_residual_route_gate().unsqueeze(1)
        return reliability_conditioned_parameters(
            self._W1,
            self._b1,
            self._W2,
            context_gain=self._gabor_freq_delta,
            view_adapter=self._gabor_view_delta,
            bias_adapter=self._gabor_phase_delta,
            hidden_gain=self._gabor_amp_delta,
            output_adapter=output_adapter,
            modulation=self._adaptive_gabor_modulation(),
            adapter_scale=self.adaptive_gabor_max_log_scale,
            view_strength=self.adaptive_gabor_view_strength,
            bias_strength=self.adaptive_gabor_phase_strength,
            output_strength=self.adaptive_gabor_residual_strength,
            view_start=2 if self.appearance_backend == "cuda_gabor8" else 0,
        )

    def get_neural_colors(self, camera_center):
        """Predict one view-conditioned RGB color for every Gaussian."""
        if self._xyz.numel() == 0:
            return torch.empty((0, self.output_dim), device=self._xyz.device)

        camera_center = camera_center.to(device=self._xyz.device, dtype=self._xyz.dtype)
        view_direction = F.normalize(
            camera_center.reshape(1, 3) - self._xyz.detach(),
            dim=-1,
            eps=1e-6,
        )
        # Bounded intrinsic descriptors keep the MLP input stable across scenes
        # without allowing appearance gradients to move or resize Gaussians.
        local_context = self._vehicle_local_coordinates()
        scale_context = torch.tanh(torch.log(self.get_scaling.detach().clamp_min(1e-6)))
        appearance_input = torch.cat((view_direction, local_context, scale_context), dim=-1)
        effective_w1, effective_b1, effective_w2 = self._conditioned_appearance_parameters()
        return neural_vehicle_color(
            appearance_input,
            effective_w1,
            effective_b1,
            effective_w2,
            self._b2,
            self.neural_output_delta_scale,
            self.neural_output_mode,
        )

    def get_cuda_gabor_contexts(self):
        """Return the extra 3 context dimensions used by cuda_gabor8.

        CUDA builds the first five inputs per pixel as (u, v, view direction).
        These detached vehicle-local coordinates provide the remaining three
        dimensions without letting appearance loss directly move Gaussians.
        """
        return self._vehicle_local_coordinates().detach().contiguous()
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors), dtype=torch.float, device="cuda").clamp(1e-4, 1.0 - 1e-4)

        W1 = torch.empty((fused_color.shape[0], self.hidden_neuron, self.input_dim), device="cuda")
        torch.nn.init.xavier_uniform_(W1)
        b1 = torch.zeros((fused_color.shape[0], self.hidden_neuron), device="cuda")
        # A zero output matrix and color-logit bias reproduce the SfM colors at
        # initialization, then allow view dependence to emerge during training.
        W2 = torch.zeros((fused_color.shape[0], self.output_dim, self.hidden_neuron), device="cuda")
        b2 = torch.logit(fused_color)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._W1 = nn.Parameter(W1.contiguous().requires_grad_(True))
        self._b1 = nn.Parameter(b1.contiguous().requires_grad_(True))
        self._W2 = nn.Parameter(W2.contiguous().requires_grad_(True))
        self._b2 = nn.Parameter(b2.contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._ensure_adaptive_gabor_state()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.configure_adaptive_gabor(training_args)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        adaptive_lr = training_args.feature_lr * float(getattr(training_args, "car_fusion_adaptive_gabor_lr_scale", 0.5))
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._W1], 'lr': training_args.feature_lr, "name": "w1"},
            {'params': [self._b1], 'lr': training_args.feature_lr, "name": "b1"},
            {'params': [self._W2], 'lr': training_args.feature_lr, "name": "w2"},
            {'params': [self._b2], 'lr': training_args.feature_lr, "name": "b2"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.adaptive_gabor_enabled:
            l.extend([
                {'params': [self._gabor_freq_delta], 'lr': adaptive_lr, "name": "gabor_freq_delta"},
                {'params': [self._gabor_view_delta], 'lr': adaptive_lr, "name": "gabor_view_delta"},
                {'params': [self._gabor_phase_delta], 'lr': adaptive_lr, "name": "gabor_phase_delta"},
                {'params': [self._gabor_amp_delta], 'lr': adaptive_lr, "name": "gabor_amp_delta"},
                {'params': [self._gabor_w2_residual], 'lr': adaptive_lr, "name": "gabor_w2_residual"},
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._W1.shape[1] * self._W1.shape[2]):
            l.append('w1_{}'.format(i))
        for i in range(self._b1.shape[1]):
            l.append('b1_{}'.format(i))
        for i in range(self._W2.shape[1] * self._W2.shape[2]):
            l.append('w2_{}'.format(i))
        for i in range(self._b2.shape[1]):
            l.append('b2_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        with torch.no_grad():
            effective_w1, effective_b1 = self.get_layer_1_weight
            effective_w2, effective_b2 = self.get_layer_2_weight
        w1 = effective_w1.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        b1 = effective_b1.detach().contiguous().cpu().numpy()
        w2 = effective_w2.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        b2 = effective_b2.detach().contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, w1, b1, w2, b2, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData(
            [el],
            comments=[
                f"appearance_model={self.APPEARANCE_MODEL_VERSION}",
                f"appearance_backend={self.appearance_backend}",
                f"neural_output_mode={self.neural_output_mode}",
                f"neural_output_delta_scale={self.neural_output_delta_scale}",
            ],
        ).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        expected_comment = f"appearance_model={self.APPEARANCE_MODEL_VERSION}"
        if expected_comment not in plydata.comments:
            raise ValueError(
                "The selected PLY uses an incompatible legacy appearance model. "
                "Render a checkpoint trained with the current implementation."
            )
        for comment in plydata.comments:
            if comment.startswith("appearance_backend="):
                self.appearance_backend = comment.split("=", 1)[1]
            if comment.startswith("neural_output_mode="):
                self.neural_output_mode = comment.split("=", 1)[1]
            if comment.startswith("neural_output_delta_scale="):
                self.neural_output_delta_scale = float(comment.split("=", 1)[1])

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        w1 = np.zeros((xyz.shape[0], self.hidden_neuron, self.input_dim))
        for i in range(self.hidden_neuron):
            for j in range(self.input_dim):
                w1[:, i, j] = np.asarray(plydata.elements[0][f"w1_{i * self.input_dim + j}"])
        b1 = np.zeros((xyz.shape[0], self.hidden_neuron))
        for i in range(self.hidden_neuron):
            b1[:, i] = np.asarray(plydata.elements[0][f"b1_{i}"])
        w2 = np.zeros((xyz.shape[0], self.output_dim ,self.hidden_neuron))
        for i in range(self.output_dim):
            for j in range(self.hidden_neuron):
                w2[:, i, j] = np.asarray(plydata.elements[0][f"w2_{i * self.hidden_neuron + j}"])
        b2 = np.zeros((xyz.shape[0], self.output_dim))
        for i in range(self.output_dim):
            b2[:, i] = np.asarray(plydata.elements[0][f"b2_{i}"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._W1 = nn.Parameter(torch.tensor(w1, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._W2 = nn.Parameter(torch.tensor(w2, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._ensure_adaptive_gabor_state()

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._W1 = optimizable_tensors["w1"]
        self._b1 = optimizable_tensors["b1"]
        self._W2 = optimizable_tensors["w2"]
        self._b2 = optimizable_tensors["b2"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "gabor_freq_delta" in optimizable_tensors:
            self._gabor_freq_delta = optimizable_tensors["gabor_freq_delta"]
        if "gabor_view_delta" in optimizable_tensors:
            self._gabor_view_delta = optimizable_tensors["gabor_view_delta"]
        if "gabor_phase_delta" in optimizable_tensors:
            self._gabor_phase_delta = optimizable_tensors["gabor_phase_delta"]
        if "gabor_amp_delta" in optimizable_tensors:
            self._gabor_amp_delta = optimizable_tensors["gabor_amp_delta"]
        if "gabor_w2_residual" in optimizable_tensors:
            self._gabor_w2_residual = optimizable_tensors["gabor_w2_residual"]
        if self._gabor_reliability.numel() == valid_points_mask.numel():
            self._gabor_reliability = self._gabor_reliability[valid_points_mask]
        if self._gabor_detail_confidence.numel() == valid_points_mask.numel():
            self._gabor_detail_confidence = self._gabor_detail_confidence[valid_points_mask]
        if self._gabor_tsdf_confidence.numel() == valid_points_mask.numel():
            self._gabor_tsdf_confidence = self._gabor_tsdf_confidence[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def finite_points_mask(self):
        if self._xyz.numel() == 0:
            return torch.zeros((0,), device="cuda", dtype=torch.bool)

        mask = torch.isfinite(self._xyz).all(dim=1)
        mask = mask & torch.isfinite(self._W1.flatten(start_dim=1)).all(dim=1)
        mask = mask & torch.isfinite(self._b1).all(dim=1)
        mask = mask & torch.isfinite(self._W2.flatten(start_dim=1)).all(dim=1)
        mask = mask & torch.isfinite(self._b2).all(dim=1)
        mask = mask & torch.isfinite(self._opacity).all(dim=1)
        mask = mask & torch.isfinite(self._scaling).all(dim=1)
        mask = mask & torch.isfinite(self._rotation).all(dim=1)
        return mask

    def prune_invalid_points(self):
        if self._xyz.numel() == 0:
            return 0
        finite_mask = self.finite_points_mask()
        invalid_mask = ~finite_mask
        invalid_count = int(invalid_mask.sum().item())
        if invalid_count == 0:
            return 0
        if invalid_count >= int(self._xyz.shape[0]):
            with torch.no_grad():
                self._xyz.data = torch.nan_to_num(self._xyz.data, nan=0.0, posinf=0.0, neginf=0.0)
                self._W1.data = torch.nan_to_num(self._W1.data, nan=0.0, posinf=0.0, neginf=0.0)
                self._b1.data = torch.nan_to_num(self._b1.data, nan=0.0, posinf=0.0, neginf=0.0)
                self._W2.data = torch.nan_to_num(self._W2.data, nan=0.0, posinf=0.0, neginf=0.0)
                self._b2.data = torch.nan_to_num(self._b2.data, nan=0.0, posinf=0.0, neginf=0.0)
                self._opacity.data = torch.nan_to_num(self._opacity.data, nan=-13.8155, posinf=13.8155, neginf=-13.8155)
                self._scaling.data = torch.nan_to_num(self._scaling.data, nan=-6.0, posinf=2.0, neginf=-6.0)
                self._rotation.data = torch.nan_to_num(self._rotation.data, nan=0.0, posinf=0.0, neginf=0.0)
            return invalid_count
        self.prune_points(invalid_mask)
        return invalid_count

    def enforce_max_primitives(self, max_primitive_num):
        self.prune_invalid_points()
        if max_primitive_num is None or self.get_xyz.shape[0] <= max_primitive_num:
            return 0

        num_points = self.get_xyz.shape[0]
        keep = int(max_primitive_num)
        opacities = self.get_opacity.squeeze()
        keep_indices = torch.topk(opacities, k=keep, largest=True, sorted=False).indices
        keep_mask = torch.zeros(num_points, device=self._xyz.device, dtype=torch.bool)
        keep_mask[keep_indices] = True
        self.prune_points(~keep_mask)
        return num_points - keep

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_w1,
        new_b1,
        new_w2,
        new_b2,
        new_opacities,
        new_scaling,
        new_rotation,
        new_gabor_freq_delta=None,
        new_gabor_amp_delta=None,
        new_gabor_w2_residual=None,
        new_gabor_reliability=None,
        new_gabor_tsdf_confidence=None,
        new_gabor_view_delta=None,
        new_gabor_phase_delta=None,
        new_gabor_detail_confidence=None,
    ):
        d = {"xyz": new_xyz,
        "w1": new_w1,
        "b1": new_b1,
        "w2": new_w2,
        "b2": new_b2,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.adaptive_gabor_enabled:
            self._ensure_adaptive_gabor_state()
            if new_gabor_freq_delta is None:
                new_gabor_freq_delta = torch.zeros((new_xyz.shape[0], self.hidden_neuron), device=new_xyz.device)
            if new_gabor_view_delta is None:
                new_gabor_view_delta = torch.zeros((new_xyz.shape[0], self.hidden_neuron), device=new_xyz.device)
            if new_gabor_phase_delta is None:
                new_gabor_phase_delta = torch.zeros((new_xyz.shape[0], self.hidden_neuron), device=new_xyz.device)
            if new_gabor_amp_delta is None:
                new_gabor_amp_delta = torch.zeros((new_xyz.shape[0], self.hidden_neuron), device=new_xyz.device)
            if new_gabor_w2_residual is None:
                new_gabor_w2_residual = torch.zeros((new_xyz.shape[0], self.output_dim, self.hidden_neuron), device=new_xyz.device)
            if new_gabor_reliability is None:
                new_gabor_reliability = torch.full((new_xyz.shape[0], 1), 0.5, device=new_xyz.device)
            if new_gabor_detail_confidence is None:
                new_gabor_detail_confidence = torch.full((new_xyz.shape[0], 1), 0.5, device=new_xyz.device)
            if new_gabor_tsdf_confidence is None:
                default_tsdf_confidence = min(max(float(self.adaptive_gabor_tsdf_default_confidence), 0.0), 1.0)
                new_gabor_tsdf_confidence = torch.full((new_xyz.shape[0], 1), default_tsdf_confidence, device=new_xyz.device)
            d["gabor_freq_delta"] = new_gabor_freq_delta
            d["gabor_view_delta"] = new_gabor_view_delta
            d["gabor_phase_delta"] = new_gabor_phase_delta
            d["gabor_amp_delta"] = new_gabor_amp_delta
            d["gabor_w2_residual"] = new_gabor_w2_residual

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._W1 = optimizable_tensors["w1"]
        self._b1 = optimizable_tensors["b1"]
        self._W2 = optimizable_tensors["w2"]
        self._b2 = optimizable_tensors["b2"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "gabor_freq_delta" in optimizable_tensors:
            self._gabor_freq_delta = optimizable_tensors["gabor_freq_delta"]
        if "gabor_view_delta" in optimizable_tensors:
            self._gabor_view_delta = optimizable_tensors["gabor_view_delta"]
        if "gabor_phase_delta" in optimizable_tensors:
            self._gabor_phase_delta = optimizable_tensors["gabor_phase_delta"]
        if "gabor_amp_delta" in optimizable_tensors:
            self._gabor_amp_delta = optimizable_tensors["gabor_amp_delta"]
        if "gabor_w2_residual" in optimizable_tensors:
            self._gabor_w2_residual = optimizable_tensors["gabor_w2_residual"]
        if self.adaptive_gabor_enabled:
            self._gabor_reliability = torch.cat((self._gabor_reliability, new_gabor_reliability.detach()), dim=0)
            self._gabor_detail_confidence = torch.cat((self._gabor_detail_confidence, new_gabor_detail_confidence.detach()), dim=0)
            self._gabor_tsdf_confidence = torch.cat((self._gabor_tsdf_confidence, new_gabor_tsdf_confidence.detach()), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.finite_points_mask())
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_w1 = self._W1[selected_pts_mask].repeat(N, 1, 1)
        new_b1 = self._b1[selected_pts_mask].repeat(N, 1)
        new_w2 = self._W2[selected_pts_mask].repeat(N, 1, 1)
        new_b2 = self._b2[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_gabor_freq_delta = None
        new_gabor_view_delta = None
        new_gabor_phase_delta = None
        new_gabor_amp_delta = None
        new_gabor_w2_residual = None
        new_gabor_reliability = None
        new_gabor_detail_confidence = None
        new_gabor_tsdf_confidence = None
        if self.adaptive_gabor_enabled:
            self._ensure_adaptive_gabor_state()
            new_gabor_freq_delta = self._gabor_freq_delta[selected_pts_mask].repeat(N, 1)
            new_gabor_view_delta = self._gabor_view_delta[selected_pts_mask].repeat(N, 1)
            new_gabor_phase_delta = self._gabor_phase_delta[selected_pts_mask].repeat(N, 1)
            new_gabor_amp_delta = self._gabor_amp_delta[selected_pts_mask].repeat(N, 1)
            new_gabor_w2_residual = self._gabor_w2_residual[selected_pts_mask].repeat(N, 1, 1)
            new_gabor_reliability = self._gabor_reliability[selected_pts_mask].repeat(N, 1)
            new_gabor_detail_confidence = self._gabor_detail_confidence[selected_pts_mask].repeat(N, 1)
            new_gabor_tsdf_confidence = self._gabor_tsdf_confidence[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_w1,
            new_b1,
            new_w2,
            new_b2,
            new_opacity,
            new_scaling,
            new_rotation,
            new_gabor_freq_delta,
            new_gabor_amp_delta,
            new_gabor_w2_residual,
            new_gabor_reliability,
            new_gabor_tsdf_confidence,
            new_gabor_view_delta,
            new_gabor_phase_delta,
            new_gabor_detail_confidence,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.finite_points_mask())
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_w1 = self._W1[selected_pts_mask]
        new_b1 = self._b1[selected_pts_mask]
        new_w2 = self._W2[selected_pts_mask]
        new_b2 = self._b2[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_gabor_freq_delta = None
        new_gabor_view_delta = None
        new_gabor_phase_delta = None
        new_gabor_amp_delta = None
        new_gabor_w2_residual = None
        new_gabor_reliability = None
        new_gabor_detail_confidence = None
        new_gabor_tsdf_confidence = None
        if self.adaptive_gabor_enabled:
            self._ensure_adaptive_gabor_state()
            new_gabor_freq_delta = self._gabor_freq_delta[selected_pts_mask]
            new_gabor_view_delta = self._gabor_view_delta[selected_pts_mask]
            new_gabor_phase_delta = self._gabor_phase_delta[selected_pts_mask]
            new_gabor_amp_delta = self._gabor_amp_delta[selected_pts_mask]
            new_gabor_w2_residual = self._gabor_w2_residual[selected_pts_mask]
            new_gabor_reliability = self._gabor_reliability[selected_pts_mask]
            new_gabor_detail_confidence = self._gabor_detail_confidence[selected_pts_mask]
            new_gabor_tsdf_confidence = self._gabor_tsdf_confidence[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_w1,
            new_b1,
            new_w2,
            new_b2,
            new_opacities,
            new_scaling,
            new_rotation,
            new_gabor_freq_delta,
            new_gabor_amp_delta,
            new_gabor_w2_residual,
            new_gabor_reliability,
            new_gabor_tsdf_confidence,
            new_gabor_view_delta,
            new_gabor_phase_delta,
            new_gabor_detail_confidence,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_primitive_num=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        hard_cap_pruned = self.enforce_max_primitives(max_primitive_num)

        torch.cuda.empty_cache()
        return {"hard_cap_pruned": hard_cap_pruned, "points": self.get_xyz.shape[0]}

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def _detail_routed_densify_score(
        self,
        errors,
        reliability=None,
        detail_weight=0.0,
        reliability_power=1.0,
        min_reliability=0.0,
    ):
        score = errors.float().reshape(-1).clone()
        if reliability is None:
            rel = torch.ones_like(score)
        else:
            rel = reliability.float().reshape(-1).to(score.device).clamp(0.0, 1.0)
            if rel.shape[0] != score.shape[0]:
                n = min(rel.shape[0], score.shape[0])
                rel_full = torch.ones_like(score)
                rel_full[:n] = rel[:n]
                rel = rel_full
        detail_weight = max(float(detail_weight), 0.0)
        reliability_power = max(float(reliability_power), 0.0)
        min_reliability = min(max(float(min_reliability), 0.0), 1.0)
        rel_gate = rel.pow(reliability_power)
        score = score * (1.0 + detail_weight * rel_gate)
        if min_reliability > 0:
            score = torch.where(rel >= min_reliability, score, torch.zeros_like(score))
        return score, rel

    def _selected_detail_strength(self, score, final_mask):
        if not final_mask.any():
            return torch.empty((0, 1), device=score.device)
        selected_score = score[final_mask].float()
        strength = selected_score / selected_score.max().clamp_min(1e-6)
        return strength.clamp(0.0, 1.0).unsqueeze(-1)

    def densify_and_split_for_error_based(
        self,
        errors,
        error_threshold,
        max_primitive_num: int,
        scene_extent,
        N=2,
        reliability=None,
        detail_weight=0.0,
        reliability_power=1.0,
        min_reliability=0.0,
        scale_shrink=0.0,
    ):
        """
        Split high-error Gaussians based on rendering error threshold.
        
        Args:
            errors: Rendering error for each Gaussian
            error_threshold: Threshold for selecting points to split
            max_primitive_num: Maximum allowed number of Gaussians
            scene_extent: Spatial extent of the scene
            N: Number of splits per selected Gaussian (default: 2)
        """
        n_init_points = self.get_xyz.shape[0]
        max_new_points = int(n_init_points * 0.05)  # Limit growth to 5% of initial points

        if max_primitive_num - n_init_points <= 0: 
            return
        max_new_points = min(max_new_points, max_primitive_num - n_init_points)

        # Extract points exceeding error threshold with large scales
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:errors.shape[0]] = errors.squeeze()
        padded_reliability = None
        if reliability is not None:
            padded_reliability = torch.ones((n_init_points), device="cuda")
            padded_reliability[:reliability.shape[0]] = reliability.squeeze()
        score, _ = self._detail_routed_densify_score(
            padded_grad,
            padded_reliability,
            detail_weight=detail_weight,
            reliability_power=reliability_power,
            min_reliability=min_reliability,
        )
        selected_pts_mask = torch.where(score >= error_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.finite_points_mask())
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # Sort by error in descending order
        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze()
        if selected_indices.numel() > 0:
            selected_errors = score[selected_indices]
            sorted_indices = selected_indices[torch.argsort(selected_errors, descending=True)]
            # Enforce limit on new points
            if sorted_indices.numel() > max_new_points:
                sorted_indices = sorted_indices[:max_new_points]
            final_mask = torch.zeros_like(selected_pts_mask)
            final_mask[sorted_indices] = True
        else:
            final_mask = selected_pts_mask  # No points selected

        detail_strength = self._selected_detail_strength(score, final_mask)
        stds = self.get_scaling[final_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[final_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[final_mask].repeat(N, 1)
        if detail_strength.numel() > 0:
            shrink = (1.0 + max(float(scale_shrink), 0.0) * detail_strength).repeat(N, 1)
        else:
            shrink = torch.ones((0, 1), device=self._xyz.device)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[final_mask].repeat(N,1) / (0.8 * N * shrink))
        new_rotation = self._rotation[final_mask].repeat(N,1)
        new_w1 = self._W1[final_mask].repeat(N, 1, 1)
        new_b1 = self._b1[final_mask].repeat(N, 1)
        new_w2 = self._W2[final_mask].repeat(N, 1, 1)
        new_b2 = self._b2[final_mask].repeat(N, 1)
        new_opacity = self._opacity[final_mask].repeat(N,1)
        new_gabor_freq_delta = None
        new_gabor_view_delta = None
        new_gabor_phase_delta = None
        new_gabor_amp_delta = None
        new_gabor_w2_residual = None
        new_gabor_reliability = None
        new_gabor_detail_confidence = None
        new_gabor_tsdf_confidence = None
        if self.adaptive_gabor_enabled:
            self._ensure_adaptive_gabor_state()
            new_gabor_freq_delta = self._gabor_freq_delta[final_mask].repeat(N, 1)
            new_gabor_view_delta = self._gabor_view_delta[final_mask].repeat(N, 1)
            new_gabor_phase_delta = self._gabor_phase_delta[final_mask].repeat(N, 1)
            new_gabor_amp_delta = self._gabor_amp_delta[final_mask].repeat(N, 1)
            new_gabor_w2_residual = self._gabor_w2_residual[final_mask].repeat(N, 1, 1)
            new_gabor_reliability = self._gabor_reliability[final_mask].repeat(N, 1)
            new_gabor_detail_confidence = torch.maximum(
                self._gabor_detail_confidence[final_mask],
                detail_strength.to(self._gabor_detail_confidence.device)
            ).repeat(N, 1)
            new_gabor_tsdf_confidence = self._gabor_tsdf_confidence[final_mask].repeat(N, 1)

        # Opacity Correction
        new_opacity = self.inverse_opacity_activation(
            1.0 - torch.sqrt(1.0 - self.opacity_activation(new_opacity))
        )

        self.densification_postfix(
            new_xyz,
            new_w1,
            new_b1,
            new_w2,
            new_b2,
            new_opacity,
            new_scaling,
            new_rotation,
            new_gabor_freq_delta,
            new_gabor_amp_delta,
            new_gabor_w2_residual,
            new_gabor_reliability,
            new_gabor_tsdf_confidence,
            new_gabor_view_delta,
            new_gabor_phase_delta,
            new_gabor_detail_confidence,
        )

        prune_filter = torch.cat((final_mask, torch.zeros(N * final_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_for_error_based(
        self,
        errors,
        error_threshold,
        max_primitive_num: int,
        scene_extent,
        reliability=None,
        detail_weight=0.0,
        reliability_power=1.0,
        min_reliability=0.0,
        opacity_boost=0.0,
    ):
        """
        Clone high-error Gaussians based on rendering error threshold.
        
        Args:
            errors: Rendering error for each Gaussian
            error_threshold: Threshold for selecting points to clone
            max_primitive_num: Maximum allowed number of Gaussians
            scene_extent: Spatial extent of the scene
        """
        n_init_points = self.get_xyz.shape[0]
        max_new_points = int(n_init_points * 0.05)  # Limit growth to 5% of initial points

        if max_primitive_num - n_init_points <= 0: 
            return
        max_new_points = min(max_new_points, max_primitive_num - n_init_points)

        score, _ = self._detail_routed_densify_score(
            errors,
            reliability,
            detail_weight=detail_weight,
            reliability_power=reliability_power,
            min_reliability=min_reliability,
        )

        # Extract points exceeding error threshold with small scales
        selected_pts_mask = torch.where(score >= error_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.finite_points_mask())
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # Sort by error in descending order
        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze()
        if selected_indices.numel() > 0:
            selected_errors = score[selected_indices]
            sorted_indices = selected_indices[torch.argsort(selected_errors, descending=True)]
            # Enforce limit on new points
            if sorted_indices.numel() > max_new_points:
                sorted_indices = sorted_indices[:max_new_points]
            final_mask = torch.zeros_like(selected_pts_mask)
            final_mask[sorted_indices] = True
        else:
            final_mask = selected_pts_mask  # No points selected

        detail_strength = self._selected_detail_strength(score, final_mask)
        new_xyz = self._xyz[final_mask]
        new_w1 = self._W1[final_mask]
        new_b1 = self._b1[final_mask]
        new_w2 = self._W2[final_mask]
        new_b2 = self._b2[final_mask]
        new_opacities = self._opacity[final_mask]
        new_scaling = self._scaling[final_mask]
        new_rotation = self._rotation[final_mask]
        new_gabor_freq_delta = None
        new_gabor_view_delta = None
        new_gabor_phase_delta = None
        new_gabor_amp_delta = None
        new_gabor_w2_residual = None
        new_gabor_reliability = None
        new_gabor_detail_confidence = None
        new_gabor_tsdf_confidence = None
        if self.adaptive_gabor_enabled:
            self._ensure_adaptive_gabor_state()
            new_gabor_freq_delta = self._gabor_freq_delta[final_mask]
            new_gabor_view_delta = self._gabor_view_delta[final_mask]
            new_gabor_phase_delta = self._gabor_phase_delta[final_mask]
            new_gabor_amp_delta = self._gabor_amp_delta[final_mask]
            new_gabor_w2_residual = self._gabor_w2_residual[final_mask]
            new_gabor_reliability = self._gabor_reliability[final_mask]
            new_gabor_detail_confidence = torch.maximum(
                self._gabor_detail_confidence[final_mask],
                detail_strength.to(self._gabor_detail_confidence.device)
            )
            new_gabor_tsdf_confidence = self._gabor_tsdf_confidence[final_mask]

        # Opacity Correction
        with torch.no_grad():
            self._opacity.data[final_mask] = self.inverse_opacity_activation(
                1.0 - torch.sqrt(1.0 - self.opacity_activation(self._opacity.data[final_mask]))
            )
        new_opacities = self.inverse_opacity_activation(
            1.0 - torch.sqrt(1.0 - self.opacity_activation(new_opacities))
        )
        if detail_strength.numel() > 0 and opacity_boost > 0:
            alpha = self.opacity_activation(new_opacities)
            alpha = alpha + float(opacity_boost) * detail_strength.to(alpha.device) * (1.0 - alpha)
            new_opacities = self.inverse_opacity_activation(alpha.clamp(1e-6, 1.0 - 1e-6))

        self.densification_postfix(
            new_xyz,
            new_w1,
            new_b1,
            new_w2,
            new_b2,
            new_opacities,
            new_scaling,
            new_rotation,
            new_gabor_freq_delta,
            new_gabor_amp_delta,
            new_gabor_w2_residual,
            new_gabor_reliability,
            new_gabor_tsdf_confidence,
            new_gabor_view_delta,
            new_gabor_phase_delta,
            new_gabor_detail_confidence,
        )

    def error_based_densify_and_prune(
        self,
        gaussian_error,
        max_primitive_num: int,
        error_threshhold: float,
        min_opacity,
        extent,
        max_screen_size,
        gaussian_reliability=None,
        detail_weight=0.0,
        reliability_power=1.0,
        min_reliability=0.0,
        scale_shrink=0.0,
        opacity_boost=0.0,
    ):
        """
        Perform error-based densification and pruning of Gaussians.
        
        This method performs two main operations:
        1. Densification: Clones and splits high-error Gaussians
        2. Pruning: Removes low-opacity and overly large Gaussians
        
        Args:
            gaussian_error: Rendering error for each Gaussian
            max_primitive_num: Maximum allowed number of Gaussians
            error_threshhold: Threshold for error-based densification
            min_opacity: Minimum opacity threshold for pruning
            extent: Spatial extent of the scene
            max_screen_size: Maximum screen-space size threshold for pruning
        """
        # Densify by cloning and splitting high-error Gaussians
        self.densify_and_clone_for_error_based(
            gaussian_error,
            error_threshhold,
            max_primitive_num,
            extent,
            reliability=gaussian_reliability,
            detail_weight=detail_weight,
            reliability_power=reliability_power,
            min_reliability=min_reliability,
            opacity_boost=opacity_boost,
        )
        self.densify_and_split_for_error_based(
            gaussian_error,
            error_threshhold,
            max_primitive_num,
            extent,
            reliability=gaussian_reliability,
            detail_weight=detail_weight,
            reliability_power=reliability_power,
            min_reliability=min_reliability,
            scale_shrink=scale_shrink,
        )

        # Apply alternative opacity reset strategy
        with torch.no_grad():
            self._opacity.data.copy_(self.inverse_opacity_activation(
                torch.clamp(self.opacity_activation(self._opacity.data) - 0.001, min=0.0)
            ))

        # Prune points based on opacity and screen-space size
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        hard_cap_pruned = self.enforce_max_primitives(max_primitive_num)

        torch.cuda.empty_cache()
        return {"hard_cap_pruned": hard_cap_pruned, "points": self.get_xyz.shape[0]}
