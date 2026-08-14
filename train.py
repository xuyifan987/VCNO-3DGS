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

import os
import random
import time
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render, network_gui
from gaussian_renderer.error_inverse_projector import inverse_project
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model_2dgs import GaussianModel as GaussianModel2DGS
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, apply_ours_module_presets
from frequency_detection import FFTBandEnergy, local_average_filter
from utils.car_fusion_utils import (
    image_reliability_map,
    fuse_frequency_reliability_error,
    reliability_weighted_photometric_loss,
    reliability_weighted_ssim_loss,
    reliability_weighted_contrast_loss,
    reliability_weighted_structure_loss,
    flat_region_mse_loss,
    SurfacePriorMemory,
)
from utils.ngs_tsdf_prior import NGSTSDFPriorFuser
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _parse_csv_list(value, cast=float):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


def _build_tsdf_schedule(opt):
    iters = _parse_csv_list(opt.car_fusion_tsdf_iters, int)
    truncs = _parse_csv_list(opt.car_fusion_tsdf_truncs, float)
    if not iters:
        return {}
    if len(truncs) == 1 and len(iters) > 1:
        truncs = truncs * len(iters)
    if len(iters) != len(truncs):
        raise ValueError("car_fusion_tsdf_iters and car_fusion_tsdf_truncs must have the same length.")
    return dict(zip(iters, truncs))


def _scheduled_scalar(default_value, start_value, end_value, start_iter, warmup, iteration):
    if start_value < 0.0 or end_value < 0.0:
        return float(default_value)
    warmup = max(float(warmup), 1.0)
    t = (float(iteration) - float(start_iter)) / warmup
    t = min(max(t, 0.0), 1.0)
    return float(start_value) * (1.0 - t) + float(end_value) * t


def _linear_ramp(iteration, start_iter, warmup):
    start_iter = max(int(start_iter), 0)
    warmup = max(int(warmup), 1)
    if iteration <= start_iter:
        return 0.0
    return min((iteration - start_iter) / warmup, 1.0)


def _camera_key(camera):
    return getattr(camera, "image_name", None) or str(getattr(camera, "uid", id(camera)))


def _pop_hard_view(viewpoint_stack, view_error_ema, opt, iteration):
    if (
        not opt.car_fusion
        or not opt.car_fusion_hard_view_sampling
        or iteration <= opt.car_fusion_hard_view_start_iter
        or len(viewpoint_stack) <= 1
        or not view_error_ema
    ):
        return viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)), 0.0

    hard_mix = _linear_ramp(
        iteration,
        opt.car_fusion_hard_view_start_iter,
        opt.car_fusion_hard_view_warmup
    )
    hard_mix *= 1.0 - min(max(float(opt.car_fusion_hard_view_uniform_mix), 0.0), 1.0)
    if random.random() > hard_mix:
        return viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)), hard_mix

    default_error = sum(view_error_ema.values()) / max(len(view_error_ema), 1)
    weights = []
    for cam in viewpoint_stack:
        score = view_error_ema.get(_camera_key(cam), default_error)
        weights.append(max(float(score), 1e-6) ** max(float(opt.car_fusion_hard_view_power), 0.0))

    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)), hard_mix

    mean_weight = weight_sum / len(weights)
    max_weight = max(float(opt.car_fusion_hard_view_max_prob_mult), 1.0) * mean_weight
    weights = [min(w, max_weight) for w in weights]
    selected = random.choices(range(len(viewpoint_stack)), weights=weights, k=1)[0]
    return viewpoint_stack.pop(selected), hard_mix


def _update_view_error_ema(view_error_ema, viewpoint_cam, l1_value, ssim_value, opt):
    ssim_gap = max(1.0 - float(ssim_value), 0.0)
    score = float(l1_value) + float(opt.car_fusion_hard_view_ssim_weight) * ssim_gap
    key = _camera_key(viewpoint_cam)
    momentum = min(max(float(opt.car_fusion_hard_view_momentum), 0.0), 0.999)
    previous = view_error_ema.get(key, score)
    view_error_ema[key] = momentum * previous + (1.0 - momentum) * score
    return view_error_ema[key]


def _hard_view_detail_multiplier(view_error_ema, viewpoint_cam, reliability_mean, opt, iteration):
    if (
        not opt.car_fusion
        or not opt.car_fusion_hard_view_sampling
        or not view_error_ema
        or iteration <= opt.car_fusion_hard_view_start_iter
    ):
        return 1.0, 0.0, 0.0

    mean_error = sum(view_error_ema.values()) / max(len(view_error_ema), 1)
    if mean_error <= 1e-8:
        return 1.0, 0.0, 0.0

    score = view_error_ema.get(_camera_key(viewpoint_cam), mean_error)
    max_ratio = max(float(opt.car_fusion_hard_view_max_prob_mult), 1.0)
    score_ratio = min(max(float(score) / float(mean_error), 0.0), max_ratio)
    difficulty = 0.0
    if max_ratio > 1.0:
        difficulty = min(max((score_ratio - 1.0) / (max_ratio - 1.0), 0.0), 1.0)
    difficulty = difficulty ** max(float(opt.car_fusion_hard_view_detail_power), 0.0)

    min_reliability = min(max(float(opt.car_fusion_hard_view_detail_min_reliability), 0.0), 1.0)
    reliability_gate = 1.0
    if min_reliability > 0.0:
        reliability_gate = min(max(float(reliability_mean) / min_reliability, 0.0), 1.0)

    schedule = _linear_ramp(
        iteration,
        opt.car_fusion_hard_view_start_iter,
        opt.car_fusion_hard_view_warmup
    )
    boost = max(float(opt.car_fusion_hard_view_detail_boost), 0.0)
    multiplier = 1.0 + boost * schedule * difficulty * reliability_gate
    return multiplier, difficulty, reliability_gate


def _metric_adaptive_multipliers(l1_value, ssim_value, opt, iteration):
    if (
        not opt.car_fusion
        or not opt.car_fusion_metric_adaptive_loss
        or iteration <= opt.car_fusion_metric_adaptive_start_iter
    ):
        return 1.0, 1.0, 1.0, 0.0, 0.0

    l1_component = max((1.0 - float(opt.lambda_dssim)) * float(l1_value), 0.0)
    ssim_component = max(float(opt.lambda_dssim) * max(1.0 - float(ssim_value), 0.0), 0.0)
    denom = max(l1_component + ssim_component, 1e-8)
    l1_share = l1_component / denom
    ssim_share = ssim_component / denom

    target_ssim_share = min(max(float(opt.car_fusion_metric_adaptive_target_ssim_share), 0.05), 0.95)
    target_l1_share = 1.0 - target_ssim_share
    l1_deficit = max(l1_share - target_l1_share, 0.0) / max(target_ssim_share, 1e-6)
    ssim_deficit = max(ssim_share - target_ssim_share, 0.0) / max(target_l1_share, 1e-6)

    schedule = _linear_ramp(
        iteration,
        opt.car_fusion_metric_adaptive_start_iter,
        opt.car_fusion_metric_adaptive_warmup
    )
    boost = max(float(opt.car_fusion_metric_adaptive_max_boost), 0.0) * schedule
    photo_multiplier = 1.0 + boost * min(max(l1_deficit, 0.0), 1.0)
    ssim_multiplier = 1.0 + boost * min(max(ssim_deficit, 0.0), 1.0)
    structure_multiplier = 1.0 + 0.5 * boost * min(max(ssim_deficit, 0.0), 1.0)
    return photo_multiplier, ssim_multiplier, structure_multiplier, l1_share, ssim_share

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    training_wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(f"Initial point num {dataset.init_point_num}, max point num: {opt.max_primitive_num}")

    frequency_bands = [
        (0.01, 0.10),
        (0.10, 0.20),
        (0.20, 0.40)
    ]

    # Initialize FFTBandEnergy module for fast frequency analysis
    fft_band_energy = FFTBandEnergy(
        bands=frequency_bands,
        use_fastlen=True,
        use_local_contrast=True,
        ksz=17
    ).cuda()

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    use_vehicle_appearance_path = bool(
        getattr(opt, "car_fusion", False)
        or getattr(opt, "vehicle_appearance", False)
        or getattr(opt, "vehicle_allocation", False)
        or getattr(opt, "vehicle_geometry", False)
        or getattr(opt, "vehicle_full", False)
        or getattr(opt, "ours_appearance", False)
        or getattr(opt, "ours_frequency", False)
        or getattr(opt, "ours_geometry", False)
        or getattr(opt, "ours_full", False)
        or getattr(opt, "car_fusion_appearance_backend", "torch_mlp") != "torch_mlp"
    )
    appearance_backend = getattr(opt, "car_fusion_appearance_backend", "torch_mlp")
    gaussians = GaussianModel(dataset.sh_degree, appearance_backend=appearance_backend) if use_vehicle_appearance_path else GaussianModel2DGS(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    surface_prior = None
    if opt.car_fusion and opt.car_fusion_surface_prior:
        surface_prior = SurfacePriorMemory(
            downsample=opt.car_fusion_surface_prior_downsample,
            momentum=opt.car_fusion_surface_prior_momentum,
            min_reliability=opt.car_fusion_surface_prior_min_reliability,
            warmup=opt.car_fusion_surface_prior_warmup,
            start_iter=opt.car_fusion_surface_prior_start
        )
    tsdf_schedule = _build_tsdf_schedule(opt) if opt.car_fusion and opt.car_fusion_tsdf_prior else {}
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    view_error_ema = {}
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if opt.car_fusion and opt.car_fusion_adaptive_gabor:
            gaussians.set_adaptive_gabor_iteration(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam, hard_view_mix = _pop_hard_view(viewpoint_stack, view_error_ema, opt, iteration)
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        train_reliability_map = None
        reliability_mean = 0.0
        if opt.car_fusion:
            train_reliability_map = image_reliability_map(
                image.detach(),
                gt_image.detach(),
                render_pkg.get("rend_alpha", None),
                kernel_size=opt.car_fusion_reliability_kernel,
                residual_temperature=opt.car_fusion_residual_temperature,
                detail_preserve=opt.car_fusion_reliability_detail_preserve,
                edge_boost=opt.car_fusion_reliability_edge_boost
            )
            reliability_mean = train_reliability_map.mean().item()
        hard_view_detail_multiplier, hard_view_difficulty, hard_view_reliability_gate = _hard_view_detail_multiplier(
            view_error_ema,
            viewpoint_cam,
            reliability_mean,
            opt,
            iteration
        )
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        metric_photo_multiplier, metric_ssim_multiplier, metric_structure_multiplier, metric_l1_share, metric_ssim_share = _metric_adaptive_multipliers(
            Ll1.item(),
            ssim_value.item(),
            opt,
            iteration
        )
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        if opt.car_fusion and train_reliability_map is not None:
            surface_weight = train_reliability_map[None].clamp(0.0, 1.0)
            surface_weight_sum = surface_weight.sum().clamp_min(1e-6)
            normal_loss = lambda_normal * (normal_error * surface_weight).sum() / surface_weight_sum
            dist_weight = surface_weight if rend_dist.dim() == surface_weight.dim() else train_reliability_map
            dist_loss = lambda_dist * (rend_dist * dist_weight).sum() / dist_weight.sum().clamp_min(1e-6)
        else:
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
        surface_prior_loss = image.sum() * 0.0
        if surface_prior is not None and train_reliability_map is not None:
            surface_prior_loss = opt.car_fusion_surface_prior_weight * surface_prior.loss(
                viewpoint_cam,
                render_pkg["surf_depth"],
                train_reliability_map,
                iteration
            )
        neural_appearance_reg = image.sum() * 0.0
        if opt.car_fusion and opt.car_fusion_adaptive_gabor:
            neural_appearance_reg = opt.car_fusion_adaptive_gabor_reg * gaussians.adaptive_gabor_regularization()
        structure_loss = image.sum() * 0.0
        structure_weight = 0.0
        detail_photo_loss = image.sum() * 0.0
        detail_photo_weight = 0.0
        detail_ssim_loss = image.sum() * 0.0
        detail_ssim_weight = 0.0
        contrast_loss = image.sum() * 0.0
        contrast_weight = 0.0
        mse_loss = image.sum() * 0.0
        mse_weight = 0.0
        flat_mse_loss = image.sum() * 0.0
        flat_mse_weight = 0.0
        if opt.car_fusion and opt.car_fusion_mse_loss:
            mse_weight = float(opt.car_fusion_mse_weight) * _linear_ramp(
                iteration,
                opt.car_fusion_mse_start_iter,
                opt.car_fusion_mse_warmup
            )
            if mse_weight > 0:
                mse_loss = mse_weight * l2_loss(image, gt_image)
        if opt.car_fusion and opt.car_fusion_flat_mse_loss:
            flat_mse_weight = float(opt.car_fusion_flat_mse_weight) * _linear_ramp(
                iteration,
                opt.car_fusion_flat_mse_start_iter,
                opt.car_fusion_flat_mse_warmup
            )
            if flat_mse_weight > 0:
                flat_mse_loss = flat_mse_weight * flat_region_mse_loss(
                    image,
                    gt_image,
                    train_reliability_map,
                    edge_suppression=opt.car_fusion_flat_mse_edge_suppression,
                    min_weight=opt.car_fusion_flat_mse_min_weight,
                    residual_gain=opt.car_fusion_flat_mse_residual_gain,
                    residual_kernel=opt.car_fusion_flat_mse_residual_kernel,
                    reliability_power=opt.car_fusion_flat_mse_reliability_power,
                )
        if opt.car_fusion and opt.car_fusion_detail_photometric_loss and train_reliability_map is not None:
            detail_photo_weight = float(opt.car_fusion_detail_photometric_weight) * _linear_ramp(
                iteration,
                opt.car_fusion_detail_photometric_start_iter,
                opt.car_fusion_detail_photometric_warmup
            )
            detail_photo_weight *= hard_view_detail_multiplier * metric_photo_multiplier
            if detail_photo_weight > 0:
                detail_photo_loss = detail_photo_weight * reliability_weighted_photometric_loss(
                    image,
                    gt_image,
                    train_reliability_map,
                    reliability_power=opt.car_fusion_detail_photometric_reliability_power,
                    min_reliability=opt.car_fusion_detail_photometric_min_reliability,
                    edge_boost=opt.car_fusion_detail_photometric_edge_boost,
                    residual_stop=opt.car_fusion_detail_photometric_residual_stop,
                    residual_floor=opt.car_fusion_detail_photometric_residual_floor,
                    loss_type=opt.car_fusion_detail_photometric_loss_type
                )
        if opt.car_fusion and opt.car_fusion_detail_ssim_loss and train_reliability_map is not None:
            detail_ssim_weight = float(opt.car_fusion_detail_ssim_weight) * _linear_ramp(
                iteration,
                opt.car_fusion_detail_ssim_start_iter,
                opt.car_fusion_detail_ssim_warmup
            )
            detail_ssim_weight *= hard_view_detail_multiplier * metric_ssim_multiplier
            if detail_ssim_weight > 0:
                detail_ssim_loss = detail_ssim_weight * reliability_weighted_ssim_loss(
                    image,
                    gt_image,
                    train_reliability_map,
                    window_size=opt.car_fusion_detail_ssim_window,
                    reliability_power=opt.car_fusion_detail_ssim_reliability_power,
                    min_reliability=opt.car_fusion_detail_ssim_min_reliability,
                    edge_boost=opt.car_fusion_detail_ssim_edge_boost
                )
        if opt.car_fusion and opt.car_fusion_structure_loss and train_reliability_map is not None:
            structure_weight = float(opt.car_fusion_structure_weight) * _linear_ramp(
                iteration,
                opt.car_fusion_structure_start_iter,
                opt.car_fusion_structure_warmup
            )
            structure_weight *= hard_view_detail_multiplier * metric_structure_multiplier
            if structure_weight > 0:
                structure_loss = structure_weight * reliability_weighted_structure_loss(
                    image,
                    gt_image,
                    train_reliability_map,
                    scales=_parse_csv_list(opt.car_fusion_structure_scales, int),
                    reliability_power=opt.car_fusion_structure_reliability_power,
                    min_reliability=opt.car_fusion_structure_min_reliability,
                    edge_boost=opt.car_fusion_structure_edge_boost
                )
        if opt.car_fusion and opt.car_fusion_contrast_loss and train_reliability_map is not None:
            contrast_weight = float(opt.car_fusion_contrast_weight) * _linear_ramp(
                iteration,
                opt.car_fusion_contrast_start_iter,
                opt.car_fusion_contrast_warmup
            )
            contrast_weight *= hard_view_detail_multiplier * metric_ssim_multiplier
            if contrast_weight > 0:
                contrast_loss = contrast_weight * reliability_weighted_contrast_loss(
                    image,
                    gt_image,
                    train_reliability_map,
                    window_size=opt.car_fusion_contrast_window,
                    scales=_parse_csv_list(opt.car_fusion_contrast_scales, int),
                    reliability_power=opt.car_fusion_contrast_reliability_power,
                    min_reliability=opt.car_fusion_contrast_min_reliability,
                    edge_boost=opt.car_fusion_contrast_edge_boost,
                    mean_weight=opt.car_fusion_contrast_mean_weight
                )

        # loss
        total_loss = loss + mse_loss + flat_mse_loss + dist_loss + normal_loss + surface_prior_loss + neural_appearance_reg + structure_loss + detail_photo_loss + detail_ssim_loss + contrast_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            hard_view_error = None
            if opt.car_fusion and opt.car_fusion_hard_view_sampling:
                hard_view_error = _update_view_error_ema(
                    view_error_ema,
                    viewpoint_cam,
                    Ll1.item(),
                    ssim_value.item(),
                    opt
                )

            extra_training_scalars = {
                'train_loss_patches/dist_loss': ema_dist_for_log,
                'train_loss_patches/normal_loss': ema_normal_for_log,
            }
            if opt.car_fusion and train_reliability_map is not None:
                extra_training_scalars['car_fusion/reliability_mean'] = reliability_mean
                extra_training_scalars['car_fusion/surface_prior_loss'] = surface_prior_loss.item()
                if opt.car_fusion_hard_view_sampling:
                    extra_training_scalars['car_fusion/hard_view_mix'] = hard_view_mix
                    extra_training_scalars['car_fusion/hard_view_detail_multiplier'] = hard_view_detail_multiplier
                    extra_training_scalars['car_fusion/hard_view_difficulty'] = hard_view_difficulty
                    extra_training_scalars['car_fusion/hard_view_reliability_gate'] = hard_view_reliability_gate
                    if hard_view_error is not None:
                        extra_training_scalars['car_fusion/hard_view_error'] = hard_view_error
                if opt.car_fusion_metric_adaptive_loss:
                    extra_training_scalars['car_fusion/metric_photo_multiplier'] = metric_photo_multiplier
                    extra_training_scalars['car_fusion/metric_ssim_multiplier'] = metric_ssim_multiplier
                    extra_training_scalars['car_fusion/metric_structure_multiplier'] = metric_structure_multiplier
                    extra_training_scalars['car_fusion/metric_l1_share'] = metric_l1_share
                    extra_training_scalars['car_fusion/metric_ssim_share'] = metric_ssim_share
                if opt.car_fusion_structure_loss:
                    extra_training_scalars['car_fusion/structure_loss'] = structure_loss.item()
                    extra_training_scalars['car_fusion/structure_weight'] = structure_weight
                if opt.car_fusion_mse_loss:
                    extra_training_scalars['car_fusion/mse_loss'] = mse_loss.item()
                    extra_training_scalars['car_fusion/mse_weight'] = mse_weight
                if opt.car_fusion_flat_mse_loss:
                    extra_training_scalars['car_fusion/flat_mse_loss'] = flat_mse_loss.item()
                    extra_training_scalars['car_fusion/flat_mse_weight'] = flat_mse_weight
                if opt.car_fusion_detail_photometric_loss:
                    extra_training_scalars['car_fusion/detail_photometric_loss'] = detail_photo_loss.item()
                    extra_training_scalars['car_fusion/detail_photometric_weight'] = detail_photo_weight
                if opt.car_fusion_detail_ssim_loss:
                    extra_training_scalars['car_fusion/detail_ssim_loss'] = detail_ssim_loss.item()
                    extra_training_scalars['car_fusion/detail_ssim_weight'] = detail_ssim_weight
                if opt.car_fusion_contrast_loss:
                    extra_training_scalars['car_fusion/contrast_loss'] = contrast_loss.item()
                    extra_training_scalars['car_fusion/contrast_weight'] = contrast_weight
                if opt.car_fusion_adaptive_gabor:
                    extra_training_scalars['vehicle_appearance/adapter_reg'] = neural_appearance_reg.item()
                    adaptive_stats = gaussians.get_adaptive_gabor_stats()
                    extra_training_scalars.update({
                        'vehicle_appearance/confidence': adaptive_stats["confidence"],
                        'vehicle_appearance/gate': adaptive_stats["gate"],
                        'vehicle_appearance/schedule': adaptive_stats["schedule"],
                        'vehicle_appearance/detail': adaptive_stats["detail"],
                        'vehicle_appearance/context_gain': adaptive_stats["context_gain"],
                        'vehicle_appearance/view_adapter': adaptive_stats["view_adapter"],
                        'vehicle_appearance/bias_adapter': adaptive_stats["bias_adapter"],
                        'vehicle_appearance/hidden_gain': adaptive_stats["hidden_gain"],
                        'vehicle_appearance/output_adapter': adaptive_stats["output_adapter"],
                        'vehicle_appearance/output_route_gate': adaptive_stats["output_route_gate"],
                    })

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if surface_prior is not None and train_reliability_map is not None:
                surface_prior.update(viewpoint_cam, render_pkg["surf_depth"], train_reliability_map, iteration)
            if opt.car_fusion and opt.car_fusion_adaptive_gabor and train_reliability_map is not None:
                projected_train_reliability, _ = inverse_project(
                    train_reliability_map,
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    background
                )
                gaussians.update_adaptive_gabor_reliability(
                    projected_train_reliability,
                    momentum=opt.car_fusion_adaptive_gabor_reliability_momentum
                )
                detail_interval = int(max(getattr(opt, "car_fusion_adaptive_gabor_detail_interval", 4), 0))
                if detail_interval > 0 and iteration % detail_interval == 0:
                    with torch.autocast('cuda', torch.float16):
                        render_detail_maps = fft_band_energy(image.detach()[None])
                        gt_detail_maps = fft_band_energy(gt_image.detach()[None])
                        detail_error_maps = torch.abs(
                            local_average_filter(render_detail_maps) - local_average_filter(gt_detail_maps)
                        ).float()[0]
                        detail_projection = torch.amax(detail_error_maps, dim=0)
                        detail_projection = fuse_frequency_reliability_error(
                            detail_projection,
                            train_reliability_map,
                            reliability_power=opt.car_fusion_reliability_power,
                            reliability_floor=opt.car_fusion_adaptive_gabor_detail_reliability_floor,
                            adaptive_floor_strength=opt.car_fusion_adaptive_gabor_detail_adaptive_floor_strength,
                            adaptive_floor_power=opt.car_fusion_adaptive_gabor_detail_adaptive_floor_power,
                            adaptive_floor_detail_power=opt.car_fusion_adaptive_gabor_detail_adaptive_floor_detail_power,
                        )
                    projected_detail, _ = inverse_project(
                        detail_projection,
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        background
                    )
                    detail_stats = gaussians.update_adaptive_gabor_detail(
                        projected_detail,
                        momentum=opt.car_fusion_adaptive_gabor_detail_momentum
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar('vehicle_appearance/projected_detail', detail_stats["mean"], iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), extra_training_scalars)

            if iteration in tsdf_schedule:
                tsdf_trunc = tsdf_schedule[iteration]
                print(f"\n[ITER {iteration}] Building Neural-Gabor TSDF prior (trunc={tsdf_trunc})")
                tsdf_fuser = NGSTSDFPriorFuser(
                    gaussians,
                    render,
                    pipe,
                    sdf_trunc=tsdf_trunc,
                    bg_color=bg_color,
                    voxel_size=opt.car_fusion_tsdf_voxel_size,
                    depth_trunc=opt.car_fusion_tsdf_depth_trunc,
                    alpha_threshold=opt.car_fusion_tsdf_alpha_threshold,
                    bounds=opt.car_fusion_tsdf_bounds,
                    bounds_padding=opt.car_fusion_tsdf_bounds_padding,
                    bounds_quantile=opt.car_fusion_tsdf_bounds_quantile,
                    chunk_size=opt.car_fusion_tsdf_chunk_size,
                    max_voxels=opt.car_fusion_tsdf_max_voxels,
                )
                query_points, tsdf, vol_origin, voxel_size, _ = tsdf_fuser.reconstruction(scene.getTrainCameras())
                gaussians.set_tsdf(tsdf, vol_origin, voxel_size, tsdf_trunc)
                if opt.car_fusion_adaptive_gabor:
                    appearance_geometry_stats = gaussians.refresh_adaptive_gabor_tsdf_confidence()
                    if tb_writer is not None:
                        tb_writer.add_scalar("vehicle_appearance/geometry_confidence", appearance_geometry_stats["mean"], iteration)
                gaussians.release_tsdf_gpu_cache()
                torch.cuda.empty_cache()
                tsdf_dir = os.path.join(scene.model_path, "car_fusion_tsdf")
                os.makedirs(tsdf_dir, exist_ok=True)
                np.save(os.path.join(tsdf_dir, f"tsdf_{iteration}.npy"), tsdf)
                np.save(os.path.join(tsdf_dir, f"vol_origin_{iteration}.npy"), vol_origin)
                if query_points is not None and len(query_points) > 0:
                    np.save(os.path.join(tsdf_dir, f"surface_points_{iteration}.npy"), query_points)
                if tb_writer is not None:
                    tb_writer.add_scalar("car_fusion/tsdf_voxel_size", voxel_size, iteration)
                    tb_writer.add_scalar("car_fusion/tsdf_surface_points", len(query_points), iteration)
                print(f"[ITER {iteration}] TSDF prior ready: {tsdf.shape}, surface samples={len(query_points)}")

            if (
                opt.car_fusion
                and opt.car_fusion_tsdf_prior
                and getattr(gaussians, "has_tsdf", lambda: False)()
                and opt.car_fusion_tsdf_pull_interval > 0
                and iteration % opt.car_fusion_tsdf_pull_interval == 0
            ):
                tsdf_stats = gaussians.tsdf_prune_and_pull(
                    pull_weight=opt.car_fusion_tsdf_pull_weight,
                    prune=opt.car_fusion_tsdf_prune,
                    outside=opt.car_fusion_tsdf_outside,
                    opacity_limit=opt.car_fusion_tsdf_prune_opacity,
                    max_pull=opt.car_fusion_tsdf_max_pull,
                    confidence_gate=opt.car_fusion_tsdf_pull_confidence_gate,
                    min_reliability=opt.car_fusion_tsdf_pull_min_reliability,
                    min_tsdf_confidence=opt.car_fusion_tsdf_pull_min_tsdf_confidence,
                    gate_power=opt.car_fusion_tsdf_pull_gate_power,
                    detail_suppression=opt.car_fusion_tsdf_pull_detail_suppression,
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("car_fusion/tsdf_pruned", tsdf_stats["pruned"], iteration)
                    tb_writer.add_scalar("car_fusion/tsdf_pulled", tsdf_stats["pulled"], iteration)
                    tb_writer.add_scalar("car_fusion/tsdf_mean_abs_sdf", tsdf_stats["mean_abs_sdf"], iteration)
                    tb_writer.add_scalar("car_fusion/tsdf_pull_gate", tsdf_stats.get("pull_gate", 1.0), iteration)
                gaussians.release_tsdf_gpu_cache()
                torch.cuda.empty_cache()

            # Error based densification
            if iteration < opt.densify_until_iter and\
                (opt.max_primitive_num == None or (opt.max_primitive_num != None and gaussians._xyz.shape[0] < opt.max_primitive_num)):
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if not opt.car_fusion:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.opacity_cull,
                            scene.cameras_extent,
                            size_threshold,
                            opt.max_primitive_num,
                        )

                # Densification
                if opt.car_fusion and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if getattr(gaussians, "has_tsdf", lambda: False)():
                        gaussians.release_tsdf_gpu_cache()
                        torch.cuda.empty_cache()
                    # Gaussian's error calculation
                    gaussians_error_max = torch.zeros((gaussians._xyz.shape[0]), device="cuda")
                    gaussians_reliability_max = torch.zeros((gaussians._xyz.shape[0]), device="cuda")
                    gaussians_view_support = torch.zeros((gaussians._xyz.shape[0]), device="cuda")
                    gaussians_error_sum = torch.zeros((gaussians._xyz.shape[0]), device="cuda")
                    gaussians_error_sq_sum = torch.zeros((gaussians._xyz.shape[0]), device="cuda")
                    gaussians_view_dir_sum = torch.zeros((gaussians._xyz.shape[0], 3), device="cuda")
                    render_view_list = scene.getTrainCameras().copy()
                    render_view_list = random.sample(render_view_list, len(render_view_list))
                    selected_render_view_list = render_view_list[:dataset.densification_camera_num] if dataset.densification_camera_num <= len(render_view_list) else render_view_list
                    densify_reliability_floor = 0.0
                    if opt.car_fusion and opt.car_fusion_densify_reliability_error and not opt.car_fusion_densify_raw_error:
                        densify_reliability_floor = _scheduled_scalar(
                            opt.car_fusion_densify_reliability_floor,
                            opt.car_fusion_densify_reliability_floor_start,
                            opt.car_fusion_densify_reliability_floor_end,
                            opt.car_fusion_densify_reliability_floor_start_iter,
                            opt.car_fusion_densify_reliability_floor_warmup,
                            iteration,
                        )
                        densify_reliability_floor = min(max(float(densify_reliability_floor), 0.0), 1.0)
                        if tb_writer is not None:
                            tb_writer.add_scalar("car_fusion/densify_reliability_floor", densify_reliability_floor, iteration)
                    for render_view in selected_render_view_list:
                        render_out = render(render_view, gaussians, pipe, background)
                        image = render_out["render"]
                        gt_image = render_view.original_image.cuda()
                        reliability_map = None
                        if opt.car_fusion:
                            reliability_map = image_reliability_map(
                                image,
                                gt_image,
                                render_out.get("rend_alpha", None),
                                kernel_size=opt.car_fusion_reliability_kernel,
                                residual_temperature=opt.car_fusion_residual_temperature,
                                detail_preserve=opt.car_fusion_reliability_detail_preserve,
                                edge_boost=opt.car_fusion_reliability_edge_boost
                            )

                        with torch.autocast('cuda', torch.float16):
                            render_image_maps = fft_band_energy(image[None])         # (1, K, H, W)
                            gt_image_maps = fft_band_energy(gt_image[None])          # (1, K, H, W)
                            render_image_maps_ave = local_average_filter(render_image_maps)
                            gt_image_maps_ave = local_average_filter(gt_image_maps)
                            error_map = torch.abs(render_image_maps_ave - gt_image_maps_ave).float()[0]
                            view_error_max = torch.zeros_like(gaussians_error_max)
                            view_effect_max = torch.zeros_like(gaussians_error_max)

                            for k in range(len(frequency_bands)):
                                projection_error = error_map[k]
                                if (
                                    opt.car_fusion
                                    and opt.car_fusion_densify_reliability_error
                                    and not opt.car_fusion_densify_raw_error
                                ):
                                    projection_error = fuse_frequency_reliability_error(
                                        projection_error,
                                        reliability_map,
                                        reliability_power=opt.car_fusion_reliability_power,
                                        reliability_floor=densify_reliability_floor,
                                        adaptive_floor_strength=opt.car_fusion_densify_adaptive_floor_strength,
                                        adaptive_floor_power=opt.car_fusion_densify_adaptive_floor_power,
                                        adaptive_floor_detail_power=opt.car_fusion_densify_adaptive_floor_detail_power,
                                    )
                                gaussians_error, gaussians_effect = inverse_project(projection_error, render_view, gaussians, pipe, background)
                                gaussians_error_max = torch.max(gaussians_error_max, gaussians_error)
                                view_error_max = torch.max(view_error_max, gaussians_error)
                                view_effect_max = torch.max(view_effect_max, gaussians_effect)
                            if opt.car_fusion:
                                projected_reliability, _ = inverse_project(reliability_map, render_view, gaussians, pipe, background)
                                gaussians_reliability_max = torch.max(gaussians_reliability_max, projected_reliability)
                            if opt.car_fusion and opt.car_fusion_visibility_aware_densify:
                                view_visible = view_effect_max > opt.car_fusion_densify_visibility_min_effect
                                if reliability_map is not None:
                                    view_visible = torch.logical_and(
                                        view_visible,
                                        projected_reliability > opt.car_fusion_densify_min_reliability
                                    )
                                view_visible_f = view_visible.float()
                                gaussians_view_support += view_visible_f
                                gaussians_error_sum += view_error_max * view_visible_f
                                gaussians_error_sq_sum += view_error_max.square() * view_visible_f
                                if opt.car_fusion_densify_view_coverage:
                                    view_dirs = torch.nn.functional.normalize(
                                        render_view.camera_center[None] - gaussians.get_xyz.detach(),
                                        dim=-1,
                                        eps=1e-6,
                                    )
                                    gaussians_view_dir_sum += view_dirs * view_visible_f[:, None]

                        del render_out, image, gt_image, render_image_maps, gt_image_maps, render_image_maps_ave, gt_image_maps_ave, error_map
                        if reliability_map is not None:
                            del reliability_map
                        torch.cuda.empty_cache()

                    size_threshold = None # Omit primitive reset based on screen size
                    error_threshold = opt.car_fusion_error_threshold if opt.car_fusion else 0.01
                    if opt.car_fusion and opt.car_fusion_visibility_aware_densify:
                        support = gaussians_view_support.clamp_min(1.0)
                        error_mean = gaussians_error_sum / support
                        error_var = (gaussians_error_sq_sum / support - error_mean.square()).clamp_min(0.0)
                        error_consistency = torch.exp(-error_var / (error_mean.square() + 1e-6))
                        error_consistency = error_consistency.clamp_min(opt.car_fusion_densify_consistency_floor)
                        min_views = max(1, min(int(opt.car_fusion_densify_min_views), len(selected_render_view_list)))
                        under_supported = (gaussians_view_support / float(min_views)).clamp(0.0, 1.0)
                        support_gate = torch.where(
                            gaussians_view_support >= float(min_views),
                            torch.ones_like(gaussians_view_support),
                            opt.car_fusion_densify_under_supported_scale * under_supported
                        )
                        coverage_gate = torch.ones_like(support_gate)
                        if opt.car_fusion_densify_view_coverage:
                            mean_view_dir = gaussians_view_dir_sum / support[:, None]
                            view_coverage = (1.0 - torch.linalg.norm(mean_view_dir, dim=-1)).clamp(0.0, 1.0)
                            coverage_floor = min(max(float(opt.car_fusion_densify_coverage_floor), 0.0), 1.0)
                            coverage_power = max(float(opt.car_fusion_densify_coverage_power), 0.0)
                            coverage_gate = coverage_floor + (1.0 - coverage_floor) * view_coverage.pow(coverage_power)
                            coverage_gate = torch.where(
                                gaussians_view_support >= float(min_views),
                                coverage_gate,
                                torch.ones_like(coverage_gate),
                            )
                        multiview_confidence = (support_gate * error_consistency * coverage_gate).clamp(0.0, 1.0)
                        confidence_blend = min(max(float(opt.car_fusion_densify_confidence_blend), 0.0), 1.0)
                        if confidence_blend > 0.0:
                            multiview_confidence = confidence_blend + (1.0 - confidence_blend) * multiview_confidence
                        gaussians_reliability_max = gaussians_reliability_max * multiview_confidence
                        if tb_writer is not None:
                            tb_writer.add_scalar("car_fusion/densify_view_support_mean", gaussians_view_support.mean(), iteration)
                            tb_writer.add_scalar("car_fusion/densify_multiview_confidence_mean", multiview_confidence.mean(), iteration)
                            if opt.car_fusion_densify_view_coverage:
                                tb_writer.add_scalar("car_fusion/densify_view_coverage_gate_mean", coverage_gate.mean(), iteration)
                    if opt.car_fusion and opt.car_fusion_uncertain_prune:
                        uncertain_mask = torch.logical_and(
                            gaussians_error_max > error_threshold,
                            gaussians_reliability_max < opt.car_fusion_uncertain_prune_threshold
                        )
                        if uncertain_mask.any():
                            gaussians.prune_points(uncertain_mask)
                            gaussians_error_max = gaussians_error_max[~uncertain_mask]
                    use_routed_densify = opt.car_fusion and (
                        opt.car_fusion_detail_routed_densify or opt.car_fusion_visibility_aware_densify
                    )
                    densify_stats = gaussians.error_based_densify_and_prune(
                        gaussian_error=gaussians_error_max,
                        max_primitive_num=opt.max_primitive_num,
                        error_threshhold=error_threshold,
                        min_opacity=opt.opacity_cull,
                        extent=scene.cameras_extent,
                        max_screen_size=size_threshold,
                        gaussian_reliability=gaussians_reliability_max if use_routed_densify else None,
                        detail_weight=opt.car_fusion_densify_detail_weight if opt.car_fusion and opt.car_fusion_detail_routed_densify else 0.0,
                        reliability_power=opt.car_fusion_densify_reliability_power,
                        min_reliability=opt.car_fusion_densify_min_reliability if use_routed_densify else 0.0,
                        scale_shrink=opt.car_fusion_densify_scale_shrink if opt.car_fusion and opt.car_fusion_detail_routed_densify else 0.0,
                        opacity_boost=opt.car_fusion_densify_opacity_boost if opt.car_fusion and opt.car_fusion_detail_routed_densify else 0.0,
                    )
                    if densify_stats is not None and tb_writer is not None:
                        tb_writer.add_scalar("car_fusion/hard_cap_pruned", densify_stats["hard_cap_pruned"], iteration)
                        tb_writer.add_scalar("total_points_after_densify", densify_stats["points"], iteration)
                    if densify_stats is not None and densify_stats["hard_cap_pruned"] > 0:
                        print(
                            f"\n[ITER {iteration}] Enforced max primitives: "
                            f"pruned {densify_stats['hard_cap_pruned']} -> {densify_stats['points']}"
                        )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                invalid_pruned = gaussians.prune_invalid_points() if hasattr(gaussians, "prune_invalid_points") else 0
                if invalid_pruned and tb_writer is not None:
                    tb_writer.add_scalar("train/invalid_points_pruned", invalid_pruned, iteration)
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if (iteration in saving_iterations):
                if hasattr(gaussians, "prune_invalid_points"):
                    invalid_pruned = gaussians.prune_invalid_points()
                    if invalid_pruned:
                        print(f"\n[ITER {iteration}] Pruned {invalid_pruned} invalid Gaussians before saving")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    torch.cuda.synchronize()
    training_wall_seconds = time.perf_counter() - training_wall_start
    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
    print(f"[EFFICIENCY] wall_seconds={training_wall_seconds:.3f}")
    print(
        f"[EFFICIENCY] peak_allocated_mb={peak_allocated_mb:.3f} "
        f"peak_reserved_mb={peak_reserved_mb:.3f}"
    )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, extra_scalars=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        if extra_scalars:
            for tag, value in extra_scalars.items():
                tb_writer.add_scalar(tag, value, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    enabled_modules = apply_ours_module_presets(args)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    print("Method implementation: original 2DGS baseline" if not enabled_modules and not args.car_fusion else "Method implementation: 2DGS + full neural vehicle modules")
    module_label = ", ".join(enabled_modules) if enabled_modules else ("custom-low-level" if args.car_fusion else "none")
    print("Ours modules: " + module_label)
    if not enabled_modules and not args.car_fusion:
        print("All vehicle modules are disabled: this run uses the original 2DGS SH baseline path.")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
